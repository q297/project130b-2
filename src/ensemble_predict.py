import mlflow
mlflow.set_tracking_uri(uri="http://0.0.0.0:8080")
from main import weighted_ensemble_predict, find_best_threshold
import os
import torch 

def find_files_in_directory(directory_path, extension):
    found_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith(extension):
            found_files.append(os.path.join(directory_path, filename))
    return found_files


print("\n Computing ENSEMBLE prediction...")

ensemble_preds = weighted_ensemble_predict(
    model_paths, val_losses, test_loader, device
)

        y_true = torch.cat([y for _, y in test_loader])

        # ---- Ищем лучший порог ----
        best_thr, best_acc = find_best_threshold(y_true, ensemble_preds)

        print(f"\n Best threshold: {best_thr:.3f}")
        print(f" Ensemble Accuracy (best thr): {best_acc:.4f}")

        mlflow.log_metric("ensemble_best_threshold", best_thr)
        mlflow.log_metric("ensemble_accuracy_best_thr", best_acc)
