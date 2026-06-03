import argparse
import csv
import dataclasses
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from .data.dataset_utils import create_splits
from .main import Config, make_datasets, train

EXPERIMENTS: list[dict] = [
    dict(model="voiceresnet", dataset="logmel"),
    dict(model="voiceresnet", dataset="mfcc"), 
    # dict(model="conv1d_lstm", dataset="logmel"),
    # dict(model="conv1d_lstm", dataset="mfcc"),
    # dict(model="whisper", dataset="logmel", n_state=128, n_head=4, n_layer=4, dropout=0.5),
    # dict(model="whisper", dataset="mfcc",   n_state=128, n_head=4, n_layer=4, dropout=0.5),
    # dict(model="whisper", dataset="logmel", n_state=256, n_head=4, n_layer=6, dropout=0.5),
    # dict(model="whisper", dataset="mfcc",   n_state=256, n_head=4, n_layer=6, dropout=0.5),
    # dict(model="cnn",          dataset="logmel"),
    # dict(model="cnn",          dataset="mfcc"),
    # dict(model="cnn_lstm_old", dataset="logmel"),
    # dict(model="cnn_lstm_old", dataset="mfcc"),
]

RESULTS_FILE = Path("sweep_results.csv")

_CSV_FIELDS = [
    "timestamp", "model", "dataset", "n_state", "n_head", "n_layer",
    "dropout", "lr", "batch_size", "augmentations",
    "test_loss", "test_acc", "test_f1", "test_bal_acc",
    "duration_sec", "status",
]


def _append_row(row: dict) -> None:
    write_header = not RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_sweep(experiments: list[dict] = EXPERIMENTS) -> None:
    print(f"Sweep: {len(experiments)} experiment(s)  |  results → {RESULTS_FILE.resolve()}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_seed = Config().seed
    create_splits(seed=default_seed)

    for i, overrides in enumerate(experiments, 1):
        cfg = Config(**{**dataclasses.asdict(Config()), **overrides})
        run_name = f"{cfg.model}_{cfg.dataset}"

        print(f"{'='*60}")
        print(f"[{i}/{len(experiments)}] {run_name}")
        print(f"  cfg: {overrides}")
        print(f"{'='*60}")

        t0 = time.monotonic()
        status = "ok"
        metrics: dict = {}

        try:
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

            with mlflow.start_run(run_name=run_name):
                metrics = train(cfg, train_ds, train_loader, val_loader, test_loader, device)

        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            status = f"error: {exc}"

        duration = time.monotonic() - t0

        row = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":        cfg.model,
            "dataset":      cfg.dataset,
            "n_state":      cfg.n_state,
            "n_head":       cfg.n_head,
            "n_layer":      cfg.n_layer,
            "dropout":      cfg.dropout,
            "lr":           cfg.lr,
            "batch_size":   cfg.batch_size,
            "augmentations": cfg.augmentations,
            "duration_sec": round(duration),
            "status":       status,
            **metrics,
        }
        _append_row(row)

        if status == "ok":
            print(
                f"\n  Result: loss={metrics.get('test_loss', '?'):.4f} "
                f"acc={metrics.get('test_acc', '?'):.3f} "
                f"F1={metrics.get('test_f1', '?'):.3f} "
                f"bal_acc={metrics.get('test_bal_acc', '?'):.3f} "
                f"({round(duration)}s)\n"
            )

    print(f"\nSweep done. Results saved to: {RESULTS_FILE.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all configs without running training",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    if args.dry_run:
        print(f"Dry run — {len(EXPERIMENTS)} experiment(s):\n")
        for i, overrides in enumerate(EXPERIMENTS, 1):
            cfg = Config(**{**dataclasses.asdict(Config()), **overrides})
            print(f"  [{i}] {cfg.model}_{cfg.dataset}: {overrides}")
    else:
        run_sweep()