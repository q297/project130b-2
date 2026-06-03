import hashlib
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from .speaker_dataset import pad_or_trim


def _parquet_hash(path: str, length: int = 8) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:length]


class STFT_Dataset(Dataset):
    """
    Dataset с комплексным STFT.

    Возвращает: (stft [2, F, T], label)
                channels = [real, imaginary]

    Параметры STFT совпадают с LogMel_Dataset:
      n_fft=400, hop_length=160, sr=16000.
    """

    def __init__(
        self,
        parquet_file: str,
        sample_rate: int = 16000,
        max_length: Union[int, float] = 5,
        n_fft: int = 400,
        hop_length: int = 160,
        device: Optional[torch.device] = None,
    ):
        self.data = pd.read_parquet(parquet_file).dropna().reset_index(drop=True)
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device or torch.device("cpu")

        phash = _parquet_hash(parquet_file)
        self.cache_file = (
            Path(__file__).parent
            / "assets"
            / f"stft_cache_{Path(parquet_file).stem}_{phash}.pt"
        )

        if not self.cache_file.exists():
            print("Precomputing STFTs...")
            all_stft: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            window = torch.hann_window(n_fft, device=self.device)

            for idx in range(len(self.data)):
                audio_path = self.data.iloc[idx, 0]
                label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

                dec = AudioDecoder(audio_path, sample_rate=sample_rate, num_channels=1)
                signal = dec.get_all_samples().data.to(self.device)  # [1, T]
                signal = pad_or_trim(signal, self.max_length)

                stft = torch.stft(
                    signal,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    return_complex=True,
                )

                stft = stft[..., :-1]

                stft_ri = torch.stack(
                    [stft.real.squeeze(0), stft.imag.squeeze(0)], dim=0
                )

                all_stft.append(stft_ri.cpu())
                all_labels.append(label)

            self.all_stft = torch.stack(all_stft)  # [N, 2, F, T]
            self.all_labels = torch.stack(all_labels)  # [N]

            torch.save(
                {"stft": self.all_stft, "labels": self.all_labels},
                self.cache_file,
            )
            print(f"Cached {len(self.all_labels)} STFTs → {self.cache_file}")
        else:
            print("Loading precomputed STFTs from disk...")
            cache = torch.load(self.cache_file, weights_only=True)
            self.all_stft = cache["stft"]
            self.all_labels = cache["labels"]

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        return self.all_stft[idx], self.all_labels[idx]
