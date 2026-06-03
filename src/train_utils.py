import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from typing import Tuple

from src.data.speaker_dataset import LogMel_Dataset, MFCC_Dataset


def train_one_epoch(model, dataloader, optimizer, loss_fn, device) -> Tuple[float, float, float, float]:
    model.train()
    total_loss = 0
    y_true, y_pred = [], []

    loop = tqdm(dataloader, desc="Training", leave=False)
    for wavs, labels in loop:
        wavs, labels = wavs.to(device), labels.to(device).float().unsqueeze(1)

        logits = model(wavs)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).int()
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)

    return avg_loss, acc, f1, bal_acc


def evaluate(model, dataloader, loss_fn, device) -> Tuple[float, float, float, float]:
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0

    loop = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for wavs, labels in loop:
            wavs, labels = wavs.to(device), labels.to(device).float().unsqueeze(1)

            logits = model(wavs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).int()
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc      = accuracy_score(y_true, y_pred)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)

    return avg_loss, acc, f1, bal_acc


def make_datasets(cfg, device: torch.device):
    root = cfg.data_root

    if cfg.dataset == "logmel":
        return (
            LogMel_Dataset(
                f"{root}/train_dataset.parquet",
                augmentations=cfg.augmentations,
                max_length=cfg.max_length,
                device=device,
                n_mels=cfg.n_mels
            ),
            LogMel_Dataset(
                f"{root}/val_dataset.parquet",
                augmentations=False,
                max_length=cfg.max_length,
                device=device,
                n_mels=cfg.n_mels
            ),
            LogMel_Dataset(
                f"{root}/test_dataset.parquet",
                augmentations=False,
                max_length=cfg.max_length,
                device=device,
                n_mels=cfg.n_mels
            ),
        )

    if cfg.dataset == "mfcc":
        return (
            MFCC_Dataset(
                f"{root}/train_dataset.parquet",
                augmentations=cfg.augmentations,
                n_mfcc=cfg.n_mels,
                max_length=cfg.max_length,
            ),
            MFCC_Dataset(
                f"{root}/val_dataset.parquet",
                augmentations=False,
                n_mfcc=cfg.n_mels,
                max_length=cfg.max_length,
            ),
            MFCC_Dataset(
                f"{root}/test_dataset.parquet",
                augmentations=False,
                n_mfcc=cfg.n_mels,
                max_length=cfg.max_length,
            ),
        )

    raise ValueError(f"Unknown dataset: '{cfg.dataset}'. Choose: patch | logmel | mfcc")