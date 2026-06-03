from pathlib import Path
from typing import Mapping, Sequence, Union
import librosa.display

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio


def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_accuracy"], "b-", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r--", label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r--", label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("training_plots.png")
    print("Graphs saved to training_plots.png")


def _to_2d_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def plot_gan_history(
    history: Mapping[str, Sequence[float]],
    path: Union[str, Path] = "gan_training_plots.png",
) -> None:
    epochs = range(1, len(history["train_G"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_G"], "b-", label="Train G")
    plt.plot(epochs, history["val_G"], "r--", label="Val G")
    plt.title("Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_D"], "b-", label="Train D")
    plt.plot(epochs, history["val_D"], "r--", label="Val D")
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"GAN loss curves saved to {path}")

def save_spectrogram_tensor(
    spec: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(spec, torch.Tensor):
        tensor = spec.detach().cpu()
        while tensor.dim() > 2:
            tensor = tensor.squeeze(0)
        torch.save(tensor, path.with_suffix(".pt"))

    elif isinstance(spec, np.ndarray):
        arr = spec.squeeze()
        np.save(path.with_suffix(".npy"), arr)

def save_spectrogram_triplet(
    real_a: Union[torch.Tensor, np.ndarray],
    real_b: Union[torch.Tensor, np.ndarray],
    fake_b: Union[torch.Tensor, np.ndarray],
    path: Union[str, Path],
    title: str = "",
    sr: int = 16000,
    n_mels: int = 128,
) -> None:
    a = _to_2d_numpy(real_a)
    b = _to_2d_numpy(real_b)
    f = _to_2d_numpy(fake_b)
    vmin = float(min(a.min(), b.min(), f.min()))
    vmax = float(max(a.max(), b.max(), f.max()))

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)
    n_ticks = 6
    tick_indices = np.linspace(0, n_mels - 1, n_ticks, dtype=int)
    tick_labels = [f"{mel_freqs[i]:.0f}" for i in tick_indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, img, name in zip(
        axes, (a, b, f), ("После операции", "До операции", "Сгенерировано")
    ):
        im = ax.imshow(
            img,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def save_spectrogram(spec, spec1, path, sample_rate=16000, hop_length=160):

    a = _to_2d_numpy(spec)
    b = _to_2d_numpy(spec1)
    vmin = float(min(a.min(), b.min()))
    vmax = float(max(a.max(), b.max()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, img, name in zip(axes, (a, b), ("После операции", "До операции")):
        im = librosa.display.specshow(
            img,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(name)
    fig.subplots_adjust(right=0.88)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _logmel_to_waveform(
    logmel: torch.Tensor,
    hop_length: int = 160,
    n_mels: int = 80,
    griffin_lim_iters: int = 32,
) -> torch.Tensor:
    if logmel.dim() == 3:
        logmel = logmel.squeeze(0)  # [n_mels, T]

    n_fft = 400

    log_spec = logmel * 4.0 - 4.0
    mel_power = 10.0**log_spec  # [n_mels, T]

    filters_path = Path(__file__).parent.parent / "data" / "assets" / "mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        mel_fb = torch.from_numpy(f[f"mel_{n_mels}"]).float()  # [n_mels, n_stft]
    mel_fb_pinv = torch.linalg.pinv(mel_fb)  # [n_stft, n_mels]
    stft_power = torch.relu(mel_fb_pinv @ mel_power)  # [n_stft, T]

    griffin_lim = torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=griffin_lim_iters,
        power=2.0,  # input is power spectrum (magnitude^2)
    )
    return griffin_lim(stft_power).unsqueeze(0)  # [1, T] for torchaudio.save


def save_audio_triplet(
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    fake_b: torch.Tensor,
    directory: Union[str, Path],
    sample_rate: int = 16000,
    hop_length: int = 160,
    n_mels: int = 80,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # real_a — source (после операции), real_b — target (до операции),
    # fake_b — сгенерировано (до операции). Совпадает с порядком в save_spectrogram_triplet.
    for name, spec in (("after", real_a), ("before", real_b), ("generated", fake_b)):
        wav = _logmel_to_waveform(spec.cpu().float(), hop_length, n_mels)
        torchaudio.save(str(directory / f"{name}.wav"), wav.cpu(), sample_rate)


def save_spectrogram_grid(
    samples: Sequence[Union[torch.Tensor, np.ndarray]],
    path: Union[str, Path],
    ncols: int = 4,
    title: str = "",
) -> None:
    """Сетка спектрограмм: удобно для быстрой качественной оценки партии."""
    n = len(samples)
    if n == 0:
        return
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n:
            img = _to_2d_numpy(samples[i])
            ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"#{i}")
        ax.axis("off")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        fig.tight_layout()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
