import argparse
import inspect

import torch

from tqdm import tqdm
from typing import Union, Callable

from configs import Config
from src.model.base_model import Autoencoder


def train_one_epoch(model: Autoencoder, train_dataloader, optimizer, criterion, device="cuda", scheduler=None):
    len_dataloader = len(train_dataloader)
    losses = torch.empty(len_dataloader, device=device)
    model.train()
    with tqdm(total=len_dataloader) as prbar:
        for batch_idx, (features, _) in enumerate(train_dataloader):
            features = features.to(device, non_blocking=True)

            restored = model(features)
            loss = criterion(restored, features)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            losses[batch_idx] = loss.detach()

            if scheduler:
                scheduler.step()

            prbar.set_description(f"epoch loss: {torch.mean(losses)}")
            prbar.update(1)

    return losses


@torch.inference_mode()
def evaluate(model: Autoencoder, dataloader, criterion, device="cuda"):
    len_dataloader = len(dataloader)
    losses = torch.empty(len_dataloader, device=device)
    model.eval()
    with tqdm(total=len_dataloader) as prbar:
        for batch_idx, (features, _) in enumerate(dataloader):
            features = features.to(device, non_blocking=True)
            restored = model(features)
            loss = criterion(restored, features)
            losses[batch_idx] = loss.detach()
            prbar.set_description(f"epoch loss: {torch.mean(losses)}")
            prbar.update(1)

    return losses


def train(model: Autoencoder, optimizer, criterion, scheduler, train_loader,
          val_loader, device="cuda", num_epoches=10):
    train_losses = torch.empty(num_epoches, device=device)
    val_losses = torch.empty(num_epoches, device=device)
    for epoch in range(1, num_epoches + 1):
        epoch_train_losses = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        epoch_val_losses = evaluate(model, val_loader, criterion, device)

        train_losses[epoch - 1] = torch.mean(epoch_train_losses)
        val_losses[epoch - 1] = torch.mean(epoch_val_losses)

    return train_losses, val_losses


def init(cfg: Config, model_constructor: Callable[..., Autoencoder]):
    def construct_entity(constructor, config):
        constructor_params = inspect.getfullargspec(constructor)[1]
        entity_params = dict()
        for param in constructor_params:
            if param in config.__dict__.keys():
                entity_params[param] = getattr(config, param, None)

        return constructor(**entity_params)

    train_cfg, model_cfg = cfg.train_config, cfg.model_config
    optim_cfg, scheduler_cfg = train_cfg.optimizer_config, train_cfg.scheduler_config

    model = construct_entity(model_constructor, model_cfg)

    if optim_cfg.name == "Adam":
        optimizer_constructor = torch.optim.Adam
    elif optim_cfg.name == "AdamW":
        optimizer_constructor = torch.optim.AdamW
    elif optim_cfg.name == "SGD":
        optimizer_constructor = torch.optim.SGD
    else:
        raise NotImplementedError("This type of optimizer is not supported")
    optimizer = construct_entity(optimizer_constructor, optim_cfg)
    criterion = train_cfg.loss()

    if scheduler_cfg is not None:
        if scheduler_cfg.name == "ExponentialLR":
            scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR
        else:
            raise NotImplementedError("This type of scheduler is not supported")
        scheduler = construct_entity(scheduler_constructor, scheduler_cfg)
    else:
        scheduler = None

    return model, optimizer, criterion, scheduler


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--type", default=None, type=str, help="autoencoder architecture type")
    args = args.parse_args()

    model_constructor = None

    if args.type is None:
        raise ValueError("No autoencoder architecture type specified. Use option --type")
