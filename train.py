import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.collators import fc_collator
from pretrain import init
from src.data.dataset import get_train_dataset, get_val_dataset


def train_classifier(model, optimizer, criterion, scheduler, train_loader, val_loader, checkpoint_path,
                     device="cpu", num_epoches=10):
    val_losses = []
    val_accs = []
    best_acc_value = 0
    for epoch in range(1, num_epoches + 1):
        losses = 0
        with tqdm(total=len(train_loader)) as prbar:
            for batch_idx, (features, targets) in enumerate(train_loader):
                features, targets = features.to(device, non_blocking=True), \
                                    targets.to(device, non_blocking=True)
                preds = model(features)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                losses += loss.detach()
                prbar.set_description(f"Loss: {losses / batch_idx}")
                prbar.update(1)

        losses = 0
        accs = 0
        with torch.no_grad(), tqdm(total=len(val_loader)) as prbar:
            for batch_idx, (features, targets) in enumerate(val_loader):
                features, targets = features.to(device, non_blocking=True), \
                                    targets.to(device, non_blocking=True)
                preds = model(features)
                loss = criterion(preds, targets)
                accuracy_score = torch.mean((preds.argmax(dim=1) == targets).float())

                losses += loss.detach()
                accs += accuracy_score.detach()

                prbar.set_description(f"Loss: {losses / batch_idx}, accuracy: {accs / batch_idx}")
                prbar.update(1)

        val_losses.append(losses / len(val_loader))
        val_accs.append(accs / len(val_loader))

        if val_accs[epoch - 1] < best_acc_value:
            best_acc_value = val_accs[epoch - 1]
            torch.save(
                {"classifier": model.state_dict()},
                os.path.join(checkpoint_path, f"classifier_epoch_{epoch}.pth")
            )

    torch.save(
        {"classifier": model.state_dict()},
        os.path.join(checkpoint_path, f"classifier_final.pth")
    )

    return val_losses, val_accs


def init_classifier(encoder, config, mode="fine_tune", freeze=False):
    if mode == "fine_tune":
        if freeze:
            for param in encoder.parameters():
                param.requires_grad_(False)
        classifier = nn.Sequential(
            encoder,
            nn.Sequential(nn.Linear(config.classifier_config.input_dim, config.classifier_config.intermediate_dim),
                          nn.ReLU(),
                          nn.Dropout(0.2),
                          nn.Linear(config.classifier_config.intermediate_dim, config.classifier_config.num_classes),
                          nn.LogSoftmax(dim=1))
        )
    elif mode == "probing":
        if freeze:
            for param in encoder.parameters():
                param.requires_grad_(False)
        classifier = nn.Sequential(
            encoder,
            nn.Sequential(nn.Linear(config.classifier_config.input_dim, config.classifier_config.num_classes),
                          nn.LogSoftmax(dim=1))
        )
    else:
        raise NotImplementedError("This mode is not supported")

    return classifier



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-t", "--type", required=True, type=str, help="autoencoder architecture type")
    args.add_argument("-d", "--download", default=False, type=bool, help="download dataset")
    args.add_argument("-nw", "--num_workers", default=1, type=int, help="num workers for dataloader")
    args.add_argument("-pp", "--pretrained_path", default=None, type=str, help="Path to pretrained model")
    args.add_argument("-m", "--mode", default="probing", type=str, help="Mode: fine tuning or probing")
    args.add_argument("-fw", "--freeze", default=False, type=bool, help="Freeze model weight during tunning")
    args.add_argument("-ne", "--num_epoches", default=10, type=bool, help="Freeze model weight during tunning")
    args = args.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.pretrained_path
    if args.type == "fc":
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

    model, _, _, scheduler = init(cfg, model_constructor)
    if path is not None:
        model.load_state_dict(torch.load(path)["model"])

    model = init_classifier(model.encoder, cfg, args.mode, args.freeze).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5 if args.freeze else 3e-3)
    criterion = nn.NLLLoss()
    scheduler = scheduler if not args.freeze else None
    train_dataset = get_train_dataset(download=args.download)
    val_dataset = get_val_dataset(download=args.download)

    train_dataloader = DataLoader(train_dataset, cfg.train_config.train_batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collator, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, cfg.train_config.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collator, pin_memory=True)

    train_classifier(model, optimizer, criterion, scheduler, train_dataloader, val_dataloader, cfg.train_config.checkpoint_path,
                     device, args.num_epoches)
