from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.speaker_dataset import LogMel_Dataset


def _get_speaker(path: str) -> str:
    for part in path.split("/"):
        if part.isdigit():
            return part
    return "unknown"


def run_error_analysis(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    output_path: Path = Path("error_analysis.csv"),
) -> pd.DataFrame:
    model.eval()
    all_probs: list[float] = []
    true_labels: list[int] = []
    with torch.no_grad():
        for wavs, labels in test_loader:
            wavs = wavs.to(device)
            probs = torch.sigmoid(model(wavs)).squeeze(1)
            all_probs.extend(probs.cpu().tolist())
            true_labels.extend(labels.int().tolist())

    paths = getattr(test_loader.dataset, "data").iloc[:, 0].tolist()

    records = []
    for path, true_lab, prob in zip(paths, true_labels, all_probs):
        pred_lab = int(prob > threshold)
        correct = true_lab == pred_lab
        if correct:
            error_type = "TP" if pred_lab == 1 else "TN"
        else:
            error_type = "FP" if pred_lab == 1 else "FN"
        records.append(
            {
                "path": path,
                "speaker": _get_speaker(path),
                "true_label": true_lab,
                "pred_label": pred_lab,
                "prob": round(prob, 4),
                "correct": correct,
                "error_type": error_type,
            }
        )

    result_df = pd.DataFrame(records)
    #result_df.to_csv(output_path, index=False)

    n_total = len(result_df)
    n_errors = int((~result_df["correct"]).sum())

    print(f"\n{'=' * 60}")
    print(f"Error Analysis  |  saved → {output_path}")
    print(f"{'=' * 60}")
    print(
        f"Total: {n_total}  |  Correct: {n_total - n_errors}  "
        f"|  Errors: {n_errors} ({100 * n_errors / n_total:.1f}%)\n"
    )

    type_counts = result_df["error_type"].value_counts()
    print("Error types:")
    for etype in ["TP", "TN", "FP", "FN"]:
        count = type_counts.get(etype, 0)
        print(f"  {etype}: {count}")

    print("\nBy speaker:")
    speaker_summary = (
        result_df.groupby("speaker")
        .agg(
            total=("correct", "count"),
            correct=("correct", "sum"),
            errors=("correct", lambda x: (~x).sum()),
            fp=("error_type", lambda x: (x == "FP").sum()),
            fn=("error_type", lambda x: (x == "FN").sum()),
            mean_prob=("prob", "mean"),
        )
        .assign(error_rate=lambda d: (d["errors"] / d["total"]).round(3))
        .sort_values("error_rate", ascending=False)
    )
    print(speaker_summary.to_string())

    errors_df = result_df[~result_df["correct"]].copy()
    if not errors_df.empty:
        errors_df["confidence"] = errors_df.apply(
            lambda r: r["prob"] if r["error_type"] == "FP" else 1.0 - r["prob"],
            axis=1,
        )
        top_errors = errors_df.nlargest(10, "confidence")[
            ["speaker", "error_type", "prob", "path"]
        ]
        print("\nTop-10 most confident mistakes:")
        print(top_errors.to_string(index=False))

    print(f"\nFull results saved to: {output_path.resolve()}\n")
    return result_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Error analysis on test set")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--model",
        required=True,
        choices=["voiceresnet", "conv1d_lstm", "whisper"],
    )
    parser.add_argument(
        "--parquet",
        default="data/processed/test_dataset.parquet",
        help="Path to test parquet",
    )
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--max_length", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="error_analysis.csv")
    args = parser.parse_args()

    from .main import Config, make_model

    cfg = Config(model=args.model, n_mels=args.n_mels, max_length=args.max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = make_model(cfg)
    m.load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location=device)
    )
    m.to(device)

    run_error_analysis(
        m,
        parquet_path=args.parquet,
        device=device,
        n_mels=args.n_mels,
        max_length=args.max_length,
        threshold=args.threshold,
        output_path=Path(args.output),
    )
