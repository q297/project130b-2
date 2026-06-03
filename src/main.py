import dataclasses
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow

from src.models.CNN import CNN
from .models.VoiceResNet import VoiceResNet
from .models.WhisperClassifier import WhisperClassifier
from .train_utils import make_datasets, train_one_epoch, evaluate
from .data.dataset_utils import create_splits
from .error_analysis import run_error_analysis

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

@dataclass
class Config:
    # --- Модель ---
    model: str = "voiceresnet"  # "conv1d_lstm" | "voiceresnet" | "whisper"
    dropout: float = 0.3
    # Параметры только для model="whisper"
    n_state: int = (
        256  # размерность скрытого состояния (128=tiny, 256=small, 384=whisper-tiny)
    )
    n_head: int = 4 
    n_layer: int = 6  # число Transformer-блоков

    # --- Обучение ---
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 70
    max_patience: int = 15
    # Дисбаланс классов: pos_weight = n_neg / n_pos (авторасчёт из датасета если -1.0)
    pos_weight: float = -1.0
    # Label smoothing: мягкие метки вместо жёстких 0/1 (0.0 = выключено)
    label_smoothing: float = 0.1

    # --- Датасет ---
    dataset: str = "logmel"  # "logmel" | "mfcc"
    n_mels: int = 128
    max_length: float = 5.0  # секунды
    augmentations: bool = True

    # --- Воспроизводимость ---
    seed: int = 333
    data_root: str = "/workspaces/super-duper-dollop/data/processed"


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Voice pre/post surgery classifier")

    for f in dataclasses.fields(defaults):
        val = getattr(defaults, f.name)
        if isinstance(val, bool):
            parser.add_argument(
                f"--{f.name}",
                default=val,
                action=argparse.BooleanOptionalAction,
            )
        else:
            parser.add_argument(f"--{f.name}", type=type(val), default=val)

    return Config(**vars(parser.parse_args()))


def make_model(cfg: Config) -> nn.Module:
    if cfg.model == "voiceresnet":
        return VoiceResNet(dropout=cfg.dropout)
    if cfg.model == "whisper":
        n_ctx = int(cfg.max_length * 16000 / 160) // 2
        return WhisperClassifier(
            n_mels=cfg.n_mels,
            n_ctx=n_ctx,
            n_state=cfg.n_state,
            n_head=cfg.n_head,
            n_layer=cfg.n_layer,
            dropout=cfg.dropout,
        )
    if cfg.model == "cnn":
        return CNN(n_mfcc=cfg.n_mels, dropout=cfg.dropout)
    raise ValueError(
        f"Unknown model: '{cfg.model}'. "
        "Choose: voiceresnet | conv1d_lstm | whisper | cnn_simple | cnn | cnn_lstm_old"
    )


def train(
    cfg: Config, train_ds, train_loader, val_loader, test_loader, device: torch.device
):
    model = make_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model}  |  Parameters: {n_params:,}")

    labels = train_ds.labels if hasattr(train_ds, "labels") else train_ds.all_labels
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    pw = cfg.pos_weight if cfg.pos_weight > 0 else n_neg / n_pos
    print(f"Class balance: до={int(n_neg)}, после={int(n_pos)} | pos_weight={pw:.3f}")
    ls = cfg.label_smoothing
    if ls > 0.0 and cfg.model == "whisper":

        def loss_fn(logits, targets):
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))(
                logits, targets * (1 - ls) + ls * 0.5
            )
    else:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    mlflow.log_params(dataclasses.asdict(cfg))
    mlflow.log_param("pos_weight_used", pw)

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_path = f"best_{cfg.model}.pt"

    for epoch in range(1, cfg.max_epochs + 1):
        train_loss, train_acc, train_f1, train_bal = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc, val_f1, val_bal = evaluate(
            model, val_loader, loss_fn, device
        )
        scheduler.step(val_loss)

        mlflow.log_metrics(
            {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
                "train_f1": float(train_f1),
                "val_f1": float(val_f1),
                "train_bal_acc": float(train_bal),
                "val_bal_acc": float(val_bal),
            },
            step=epoch,
        )
        print(
            f"Epoch {epoch:3d}: "
            f"TL={train_loss:.4f} TA={train_acc:.3f} F1={train_f1:.3f} BA={train_bal:.3f} | "
            f"VL={val_loss:.4f} VA={val_acc:.3f} F1={val_f1:.3f} BA={val_bal:.3f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ Saved: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.max_patience:
                print("  ⛔ Early stopping")
                break

    print(f"\nLoading best checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_loss, test_acc, test_f1, test_bal = evaluate(
        model, test_loader, loss_fn, device
    )
    mlflow.log_metrics(
        {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1": float(test_f1),
            "test_bal_acc": float(test_bal),
        }
    )
    print(
        f"Test: loss={test_loss:.4f} acc={test_acc:.3f} F1={test_f1:.3f} bal_acc={test_bal:.3f}"
    )

    run_error_analysis(
        model,
        test_loader=test_loader,
        device=device,
        output_path=Path(f"errors_{cfg.model}_{cfg.dataset}.csv"),
    )

    return {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "test_bal_acc": float(test_bal),
    }


def main():
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {cfg.model}  |  Dataset: {cfg.dataset}")

    create_splits(seed=cfg.seed)

    train_ds, val_ds, test_ds = make_datasets(cfg, device)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    with mlflow.start_run(run_name=f"{cfg.model}_{cfg.dataset}"):
        train(cfg, train_ds, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
