from data.speaker_dataset import SpeakerDataset
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
from models.CNN_LSTM_Attention import CNN_LSTM_Attention
from models.CNN import CNN_Simple, CNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from train_utils import train_one_epoch, evaluate
import mlflow
import numpy as np
import random
from data.dataset_utils import create_splits

mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")

project_params = {
    "model": CNN,
    "max_patience": 5,
    "max_epochs": 30,
    "batch_size": 32,
    "regularization_value": 0.001,
    "dropout_value": 0.4,
    "n_mfcc": 128,
    "augmentations": True,
    "n_models": 1,
}


# ------------------------------
#  АНСАМБЛЕВОЕ ПРЕДСКАЗАНИЕ
# ------------------------------
def ensemble_predict(model_paths, test_loader, device):
    preds_all = []

    for path in model_paths:
        model = project_params["model"](dropout=project_params["dropout_value"]).to(
            device
        )
        model.load_state_dict(torch.load(path))
        model.eval()

        preds = []

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits = model(x)
                preds.append(torch.sigmoid(logits).cpu())

        preds_all.append(torch.cat(preds))

    # усреднение по моделям
    ensemble_preds = torch.mean(torch.stack(preds_all), dim=0)
    return ensemble_preds


def weighted_ensemble_predict(model_paths, val_losses, test_loader, device):
    preds_all = []

    # compute weights: w_i = 1 / val_loss_i
    weights = torch.tensor([1 / v for v in val_losses], dtype=torch.float32)
    weights = weights / weights.sum()  # normalize

    for path in model_paths:
        model = project_params["model"](dropout=project_params["dropout_value"]).to(
            device
        )
        model.load_state_dict(torch.load(path))
        model.eval()

        preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                logits = model(x)
                preds.append(torch.sigmoid(logits).cpu())

        preds_all.append(torch.cat(preds))

    preds_all = torch.stack(preds_all)  # shape: (n_models, N_samples, 1)
    weights = weights[:, None, None]  # reshape for broadcasting

    # weighted sum
    ensemble_preds = torch.sum(preds_all * weights, dim=0)
    return ensemble_preds


# ------------------------------
#  ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ
# ------------------------------
def train_single_model(model_idx, train_loader, val_loader, test_loader, device):
    model = project_params["model"](n_mfcc=project_params["n_mfcc"], dropout=project_params["dropout_value"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=project_params["regularization_value"]
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    mlflow.log_params(
        {k: (v if not callable(v) else v.__name__) for k, v in project_params.items()}
    )

    best_val_loss = float("inf")
    patience = project_params["max_patience"]
    patience_counter = 0
    ckpt_path = None

    for epoch in range(1, project_params["max_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc, roc_auc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        print(
            f"[Model {model_idx}] Epoch {epoch}: TL={train_loss:.4f}, VL={val_loss:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = f"best_model_{model.__class__.__name__.lower()}_{model_idx}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✔ Saved: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⛔ Early stopping for model {model_idx}")
                break

    # evaluate test
    test_loss, test_acc, test_auc = evaluate(model, test_loader, loss_fn, device)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_acc", test_acc)

    return ckpt_path, best_val_loss


# ------------------------------
#  MAIN
# ------------------------------
def main():
    # загрузка датасетов
    train_dataset = SpeakerDataset(
        parquet_file="/workspaces/super-duper-dollop/data/processed/train_dataset.parquet",
        n_mfcc=project_params["n_mfcc"],
        augmentations=project_params["augmentations"],
    )
    val_dataset = SpeakerDataset(
        parquet_file="/workspaces/super-duper-dollop/data/processed/val_dataset.parquet",
        n_mfcc=project_params["n_mfcc"],
        augmentations=project_params["augmentations"],
    )
    test_dataset = SpeakerDataset(
        parquet_file="/workspaces/super-duper-dollop/data/processed/test_dataset.parquet",
        n_mfcc=project_params["n_mfcc"],
        augmentations=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=project_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=project_params["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=project_params["batch_size"], shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- ОБУЧЕНИЕ НЕСКОЛЬКО МОДЕЛЕЙ ----
    model_paths = []
    val_losses = []
    seeds = [2026, 666]
    mlflow.log_param("seeds", seeds)
    for i in range(project_params["n_models"]):
        create_splits(seed=seeds[i])
        torch.manual_seed(seeds[i])
        torch.cuda.manual_seed_all(seeds[i])
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        with mlflow.start_run(run_name=f"model_{i + 1}"):
            path, val_loss = train_single_model(
                i + 1, train_loader, val_loader, test_loader, device
            )
            model_paths.append(path)
            val_losses.append(val_loss)

    if project_params["n_models"] > 1:
        # ---- АНСАМБЛЬ ----
        print("\n Computing ENSEMBLE prediction...")
        ensemble_preds = weighted_ensemble_predict(
            model_paths, val_losses, test_loader, device
        )

        # получаем y_true
        y_true = torch.cat([y for _, y in test_loader])

        # accuracy
        ensemble_acc = (ensemble_preds > 0.5).float().eq(y_true).float().mean().item()

        print(f"\n Ensemble Accuracy: {ensemble_acc:.4f}")
        mlflow.log_metric("ensemble_accuracy", ensemble_acc)


if __name__ == "__main__":
    main()
