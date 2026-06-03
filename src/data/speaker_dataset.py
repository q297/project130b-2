import hashlib
import random
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
import torchaudio
from torch_audiomentations import (
    Compose,
    AddColoredNoise,
    Shift,
    Gain,
    PolarityInversion,
    PeakNormalization,
)

def _parquet_hash(path: str, length: int = 8) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:length]


AUGMENTATIONS = Compose(
    transforms=[
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.5, output_type="dict"),
        AddColoredNoise(
            min_snr_in_db=20,
            min_f_decay=0,
            max_snr_in_db=35,
            p=0.5,
            output_type="dict",
        ),
        Shift(
            min_shift=-0.1,
            max_shift=0.1,
            rollover=False,
            p=0.3,
            output_type="dict",
        ),
        PolarityInversion(p=0.5, output_type="dict"),
        PeakNormalization(p=0.3, output_type="dict"),
    ],
    output_type="dict",
    shuffle=True,
)


def pad_or_trim(signal, max_length: int):
    """
    Pad or trim the audio array to MAX_LENGTH, as expected by the encoder.
    """
    length = signal.shape[1]
    if length > max_length:
        start = torch.randint(0, length - max_length + 1, (1,)).item()
        signal = signal[:, start : start + max_length]
    else:
        pad = max_length - length
        signal = torch.cat([signal, torch.zeros(signal.shape[0], pad, device=signal.device)], dim=1)
    return signal


def augment(signal, sample_rate: int):
    signal = signal.unsqueeze(0)  # [B=1, C=1, T=num_samples]
    signal = AUGMENTATIONS(signal, sample_rate=sample_rate).samples
    signal = signal.squeeze(0)  # [C, T] для дальнейшей обработки
    return signal


def spec_augment(
    logmel: torch.Tensor,
    freq_mask_param: int = 15,
    time_mask_param: int = 60,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment applied to a cached logmel spectrogram.

    logmel: [C, F, T]  — значения не изменяются в кэше,
    """
    logmel = logmel.clone()
    _, n_mels, n_frames = logmel.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f > 0:
            f0 = random.randint(0, max(0, n_mels - f))
            logmel[:, f0 : f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        if t > 0:
            t0 = random.randint(0, max(0, n_frames - t))
            logmel[:, :, t0 : t0 + t] = 0.0

    return logmel


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = Path(__file__).parent / "assets" / "mel_filters.npz"
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


class LogMel_Dataset(Dataset):
    """
    Dataset с логмел-спектрограммами.
    """

    def __init__(
        self,
        parquet_file: str,
        sample_rate: int = 16000,
        max_length: Union[int, float] = 3,
        device: Optional[torch.device] = None,
        n_mels: int = 80,
        augmentations: Optional[bool] = False,
    ):
        self.data = pd.read_parquet(parquet_file).dropna().reset_index(drop=True)
        self.sample_rate = sample_rate
        self.hop_length = 160
        self.max_length = int(max_length * sample_rate)
        assert self.max_length % self.hop_length == 0, (
            "max_length must be multiple of hop_length"
        )
        self.n_fft = 400
        self.device = device if device else torch.device("cpu")
        self.n_mels = n_mels
        self.augmentations = augmentations  
        phash = _parquet_hash(parquet_file)
        self.precompute_file = (
            Path(__file__).parent
            / "assets"
            / f"logmel_cache_{Path(parquet_file).stem}_{n_mels}mels_{phash}.pt"
        )

        if not self.precompute_file.exists():
            print("Precomputing LogMels...")
            all_logmel = []
            all_labels = []
            for idx in range(len(self.data)):
                audio_path = self.data.iloc[idx, 0]
                label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
                dec = AudioDecoder(
                    audio_path, sample_rate=self.sample_rate, num_channels=1,
                )
                signal = dec.get_all_samples().data  # [1, num_samples]
                signal = pad_or_trim(signal, self.max_length).to(self.device)
                window = torch.hann_window(self.n_fft, device=self.device)
                stft = torch.stft(
                    signal,
                    n_fft=self.n_fft,
                    hop_length=160,
                    window=window,
                    return_complex=True,
                )
                magnitudes = stft[..., :-1].abs() ** 2

                filters = mel_filters(self.device, n_mels)
                mel_spec = filters @ magnitudes

                log_spec = torch.clamp(mel_spec, min=1e-10).log10()
                log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
                log_spec = (log_spec + 4.0) / 4.0

                all_logmel.append(log_spec)
                all_labels.append(label)

            self.all_logmels = torch.stack(all_logmel)
            self.all_labels = torch.stack(all_labels)

            torch.save(
                {"logmels": self.all_logmels, "labels": self.all_labels},
                self.precompute_file,
            )
        else:
            print("Loading precomputed LogMels from disk...")
            cache = torch.load(self.precompute_file, weights_only=True)
            self.all_logmels = cache["logmels"]
            self.all_labels = cache["labels"]

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        logmel = self.all_logmels[idx]
        label = self.all_labels[idx]
        if self.augmentations:
            logmel = spec_augment(logmel)
        return logmel, label


class MFCC_Dataset(Dataset):
    def __init__(
        self,
        parquet_file: str,
        sample_rate: int = 16000,
        max_length: Union[int, float] = 3,
        n_mfcc: int = 32,
        augmentations: Optional[bool] = False,
    ):
        """
        Speaker dataset with TorchCodec decoding.

        Args:
            parquet_file (str): Path to the parquet file containing audio file paths and labels.
            sample_rate (int, optional): Desired sample rate for audio files. Defaults to 16000.
            max_length (int | float, optional): Max audio length in seconds
            n_mfcc (int, optional): Number of MFCC
            augmentations (bool, optional): Whether to apply data augmentations
        """
        self.data = pd.read_parquet(parquet_file).dropna().reset_index(drop=True)
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        )
        self.augmentations = augmentations
        phash = _parquet_hash(parquet_file)
        self.precompute_file = (
            Path(__file__).parent
            / "assets"
            / f"mfcc_cache_{Path(parquet_file).stem}_{n_mfcc}mfcc_{phash}.pt"
        )
        if not self.precompute_file.exists():
            print("Precomputing MFCCs...")

            all_mfcc = []
            all_labels = []

            for idx in range(len(self.data)):
                audio_path = self.data.iloc[idx, 0]
                label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

                dec = AudioDecoder(
                    audio_path, sample_rate=self.sample_rate, num_channels=1
                )
                signal = dec.get_all_samples().data
                signal = pad_or_trim(signal, self.max_length)

                # MFCC
                mfcc = self.mfcc_transform(signal).squeeze(0)

                mean = mfcc.mean(dim=1, keepdim=True)
                std = mfcc.std(dim=1, keepdim=True) + 1e-6
                mfcc_norm = (mfcc - mean) / std

                all_mfcc.append(mfcc_norm)
                all_labels.append(label)

            self.all_mfcc = torch.stack(all_mfcc)  # (N, n_mfcc, n_frames)
            self.all_labels = torch.stack(all_labels)  # (N,)

            torch.save(
                {"mfcc": self.all_mfcc, "labels": self.all_labels}, self.precompute_file
            )

        else:
            print("Loading precomputed MFCCs from disk...")
            cache = torch.load(self.precompute_file, weights_only=True)
            self.all_mfcc = cache["mfcc"]
            self.all_labels = cache["labels"]

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        mfcc = self.all_mfcc[idx] 
        if self.augmentations:
            mfcc = spec_augment(mfcc.unsqueeze(0)).squeeze(0)
        return mfcc, self.all_labels[idx]
