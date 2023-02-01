import argparse
import inspect
import os

import numpy as np
import torch
import torch.nn as nn
from IPython.core.display import clear_output
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Union, Callable

from configs import Config
from src.data.collators import fc_collator, bw_collator
from src.data.dataset import get_train_dataset, get_val_dataset, SSLDataset, BlackWhiteDataset
from src.model import Simple
from src.model.BWAE import BWAE
from src.model.GANAE import GANAE, Descriminator
from src.model.Simple import SimpleAE
from src.model.base_model import Autoencoder
from src.utils.utils import construct_grid, show_grid, CosineFocalLoss

import torchvision.transforms as T

SQ = 6*6


def train_one_epoch(model_G, model_D, train_dataloader, optimizer_G, optimizer_D, criterion,
                    criterion_visual, device="cuda", scheduler_D=None, scheduler_G=None,
                    scheduler_frequency=None, batch_size=32, loss_mix=1.):
    len_dataloader = len(train_dataloader)
    losses_D = 0
    losses_G = 0
    model_G.train()
    model_D.train()
    real_label = torch.ones((batch_size, SQ), device=device)
    fake_label = torch.zeros((batch_size, SQ), device=device)
    with tqdm(total=len_dataloader) as prbar:
        for batch_idx, (features, targets) in enumerate(train_dataloader):
            features = features.to(device, non_blocking=True)
            if targets is not None:
                targets = targets.to(device, non_blocking=True)

            optimizer_D.zero_grad()
            D_pred = model_D(targets)
            loss_D = criterion(D_pred, real_label)
            loss_D.backward()

            fake = model_G(features)
            D_pred = model_D(fake.detach())
            fake_loss_D = criterion(D_pred, fake_label)
            fake_loss_D.backward()
            loss_D += fake_loss_D
            optimizer_D.step()


            optimizer_G.zero_grad()
            D_pred = model_D(fake)
            loss_G = (1 - loss_mix) * criterion(D_pred, real_label) + loss_mix * criterion_visual(fake, targets)
            loss_G.backward()
            optimizer_G.step()

            losses_D += loss_D.detach()
            losses_G += loss_G.detach()

            if scheduler_D is not None and scheduler_frequency == "step":
                scheduler_D.step()
                scheduler_G.step()

            prbar.set_description(f"D loss: {losses_D / (batch_idx + 1)}, G loss: {losses_G / (batch_idx + 1)}")
            prbar.update(1)

    losses_D /= (batch_idx + 1)
    losses_G /= (batch_idx + 1)
    return losses_D, losses_G


@torch.inference_mode()
def evaluate(model_G, model_D, dataloader, criterion, criterion_visual, device="cuda",
             batch_size=32, loss_mix=1.):
    len_dataloader = len(dataloader)
    losses = 0
    model_D.eval()
    model_G.eval()
    real_label = torch.ones((batch_size, SQ), device=device)
    with tqdm(total=len_dataloader) as prbar:
        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device, non_blocking=True)
            if targets is not None:
                targets = targets.to(device, non_blocking=True)
            fake = model_G(features)
            D_pred = model_D(fake)
            loss_G = (1 - loss_mix) * criterion(D_pred, real_label) + loss_mix * criterion_visual(fake, targets)
            losses += loss_G.detach()
            prbar.set_description(f"epoch loss: {losses / (batch_idx + 1)}")
            prbar.update(1)

    losses /= (batch_idx + 1)
    return losses


def train(model_D, model_G, optimizer_D, optimizer_G, criterion, criterion_visual, scheduler_D, scheduler_G, train_loader, val_dataset,
          val_loader, checkpoint_path, device="cuda", num_epoches=10, scheduler_frequency=None, draw=False, batch_size=32,
          loss_mix=1.):
    train_losses_D = []
    train_losses_G = []
    val_losses = []
    best_loss_value = torch.inf
    os.makedirs(checkpoint_path, exist_ok=True)
    for epoch in range(1, num_epoches + 1):
        epoch_train_losses = train_one_epoch(model_G, model_D, train_loader, optimizer_G, optimizer_D,
                                             criterion, criterion_visual, device, scheduler_D,
                                             scheduler_G, scheduler_frequency, batch_size, loss_mix)
        epoch_val_losses = evaluate(model_G, model_D, val_loader, criterion, criterion_visual, device, batch_size, loss_mix)

        train_losses_D.append(epoch_train_losses[0])
        train_losses_G.append(epoch_train_losses[1])
        val_losses.append(epoch_val_losses)

        if draw:
            x = np.arange(epoch)
            clear_output()
            plt.subplot(1, 2, 1)
            plt.title("Generator loss")
            plt.plot(x, train_losses_G, c="C2", label="train")
            plt.plot(x, val_losses, c="C1", label="val")
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.title("Discriminator loss")
            plt.plot(x, train_losses_D, c="C2")
            plt.show()
            grid = construct_grid(model_G, val_dataset)
            show_grid(grid)
            plt.show()

        if val_losses[epoch - 1] < best_loss_value:
            best_loss_value = val_losses[epoch - 1]
            torch.save(
                {"G": model_G.state_dict(),
                 "D": model_D.state_dict()},
                os.path.join(checkpoint_path, f"epoch_{epoch}.pth")
            )

        if scheduler_D is not None and scheduler_frequency == "epoch":
            scheduler_D.step()
            scheduler_G.step()

    torch.save(
        {"G": model_G.state_dict(),
         "D": model_D.state_dict()},
        os.path.join(checkpoint_path, f"final.pth")
    )

    return train_losses_D, train_losses_G, val_losses


device = "cpu"
num_epoch = 400
num_warmaup_epoches = 30
batch_size = 1
loss_mix = .998

model_G = GANAE(bw=True)
model_D = Descriminator(bw=True)

train_dataset = BlackWhiteDataset("data/data/train/unlabeled", labled=False)
if __name__ == '__main__':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=bw_collator, num_workers=6, pin_memory=True)
    batch = bw_collator([(train_dataset[0][0],)], no_targets=True)
    # restored = model_G(batch[0])
    # model_D(restored)

    optimizerD = torch.optim.Adam(model_D.parameters(), lr=4e-3, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=4e-3, betas=(0.5, 0.999))
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, len(train_dataset) // batch_size)
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, len(train_dataset) // batch_size)

    criterion = nn.BCELoss()
    criterion_visual = nn.L1Loss()

    train(model_D, model_G, optimizerD, optimizerG, criterion, criterion_visual, schedulerD,
          schedulerG, train_loader, train_dataset, train_loader, "checkpoints/gae",
          device, num_epoch, "step", True, batch_size, loss_mix)
