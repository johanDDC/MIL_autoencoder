import argparse
import torch

from tqdm import tqdm

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


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--type", default=None, type=str, help="autoencoder architecture type")
    args = args.parse_args()

    if args.type is None:
        raise ValueError("No autoencoder architecture type specified. Use option --type")
