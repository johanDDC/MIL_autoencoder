import argparse
import inspect
import os

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Union, Callable

from configs import Config
from src.data.collators import fc_collator
from src.data.dataset import get_train_dataset, get_val_dataset
from src.model.base_model import Autoencoder


def train_one_epoch(model: Autoencoder, train_dataloader, optimizer, criterion, device="cuda", scheduler=None,
                    scheduler_frequency=None):
    len_dataloader = len(train_dataloader)
    losses = torch.zeros(len_dataloader, device=device)
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

            if scheduler is not None and scheduler_frequency=="step":
                scheduler.step()

            prbar.set_description(f"epoch loss: {torch.sum(losses) / (batch_idx + 1)}")
            prbar.update(1)

    return losses


@torch.inference_mode()
def evaluate(model: Autoencoder, dataloader, criterion, device="cuda"):
    len_dataloader = len(dataloader)
    losses = torch.zeros(len_dataloader, device=device)
    model.eval()
    with tqdm(total=len_dataloader) as prbar:
        for batch_idx, (features, _) in enumerate(dataloader):
            features = features.to(device, non_blocking=True)
            restored = model(features)
            loss = criterion(restored, features)
            losses[batch_idx] = loss.detach()
            prbar.set_description(f"epoch loss: {torch.sum(losses) / (batch_idx + 1)}")
            prbar.update(1)

    return losses


def train(model: Autoencoder, optimizer, criterion, scheduler, train_loader,
          val_loader, checkpoint_path, device="cuda", num_epoches=10, scheduler_frequency=None):
    train_losses = torch.empty(num_epoches, device=device)
    val_losses = torch.empty(num_epoches, device=device)
    best_loss_value = torch.inf
    os.makedirs(checkpoint_path, exist_ok=True)
    for epoch in range(1, num_epoches + 1):
        epoch_train_losses = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        epoch_val_losses = evaluate(model, val_loader, criterion, device)

        train_losses[epoch - 1] = torch.mean(epoch_train_losses)
        val_losses[epoch - 1] = torch.mean(epoch_val_losses)

        if val_losses[epoch - 1] < best_loss_value:
            best_loss_value = val_losses[epoch - 1]
            torch.save(
                {"model": model.state_dict()},
                os.path.join(checkpoint_path, f"epoch_{epoch}.pth")
            )

        if scheduler is not None and scheduler_frequency == "epoch":
            scheduler.step()

    torch.save(
        {"model": model.state_dict()},
        os.path.join(checkpoint_path, f"final.pth")
    )

    return train_losses, val_losses


def init(cfg: Config, model_constructor: Callable[..., Autoencoder]):
    def construct_entity(constructor, config, **kwargs):
        constructor_params = inspect.getfullargspec(constructor).args
        entity_params = dict()
        for param in constructor_params:
            if param != "self" and param in config.__dir__():
                entity_params[param] = getattr(config, param, None)

        return constructor(**entity_params, **kwargs)

    train_cfg, model_cfg = cfg.train_config, cfg.model_config
    optim_cfg, scheduler_cfg = train_cfg.optimizer_config, train_cfg.scheduler_config

    model = construct_entity(model_constructor, model_cfg)
    if model_cfg.use_pretrained:
        model.load_state_dict(torch.load(model_cfg.pretrained_path)["model"])

    if optim_cfg.name == "Adam":
        optimizer_constructor = torch.optim.Adam
    elif optim_cfg.name == "AdamW":
        optimizer_constructor = torch.optim.AdamW
    elif optim_cfg.name == "SGD":
        optimizer_constructor = torch.optim.SGD
    else:
        raise NotImplementedError("This type of optimizer is not supported")
    optimizer = construct_entity(optimizer_constructor, optim_cfg, params=model.parameters())
    criterion = train_cfg.loss()

    if scheduler_cfg is not None:
        if scheduler_cfg.name == "ExponentialLR":
            scheduler_constructor = torch.optim.lr_scheduler.ExponentialLR
        elif scheduler_cfg.name == "CosineAnnealingLR":
            scheduler_constructor = torch.optim.lr_scheduler.CosineAnnealingLR
        elif scheduler_cfg.name == "LambdaLR":
            scheduler_constructor = torch.optim.lr_scheduler.LambdaLR
        else:
            raise NotImplementedError("This type of scheduler is not supported")
        scheduler = construct_entity(scheduler_constructor, scheduler_cfg, optimizer=optimizer)
    else:
        scheduler = None

    return model, optimizer, criterion, scheduler


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--type", default=None, type=str, help="autoencoder architecture type")
    args.add_argument("-d", "--download", default=False, type=bool, help="download dataset")
    args.add_argument("-nw", "--num_workers", default=1, type=int, help="num workers for dataloader")
    args = args.parse_args()

    model_constructor = None
    cfg = None
    collator = None
    scheduler_freq = None

    if args.type is None:
        raise ValueError("No autoencoder architecture type specified. Use option --type")
    elif args.type == "fc":
        from configs import FCConfig
        from src.model.FullyConnectedAE import FCAutoencoder

        model_constructor = FCAutoencoder
        cfg = FCConfig
        collator = fc_collator
    elif args.type == "conv":
        from configs import CAEConfig
        from src.model.ConvolutionalAE import CNNAutoencoder

        model_constructor = CNNAutoencoder
        cfg = CAEConfig
    elif args.type == "mae":
        from configs import MAEConfig
        from src.model.mae.MAE import MAE

        model_constructor = MAE
        cfg = MAEConfig

    model, optimizer, criterion, scheduler = init(cfg, model_constructor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device="cpu"
    model.to(device)

    train_dataset = get_train_dataset(download=args.download)
    val_dataset = get_val_dataset(download=args.download)

    train_dataloader = DataLoader(train_dataset, cfg.train_config.train_batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collator, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, cfg.train_config.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collator, pin_memory=True)
    if scheduler is not None:
        scheduler_freq = cfg.train_config.scheduler_config.frequency

    train_losses, val_losses = train(model, optimizer, criterion, scheduler,
                                     train_dataloader, val_dataloader, cfg.train_config.checkpoint_path,
                                     device=device, num_epoches=cfg.train_config.num_epoches,
                                     scheduler_frequency=scheduler_freq)
    print("Final mean train loss:", train_losses.mean())
    print("Final mean val loss:", val_losses.mean())
