import hashlib
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder

from .speaker_dataset import mel_filters, pad_or_trim, spec_augment


def _file_hash(path: str, length: int = 8) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:length]


def _load_pairs(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".ods":
        df = pd.read_excel(path, engine="odf")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported pairs file format: {suffix}")

    required = {"before", "after", "speaker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairs file missing columns: {missing}")
    return df.dropna().reset_index(drop=True)


class PairedLogMel_Dataset(Dataset):

    def __init__(
        self,
        pairs_file: str,
        data_root: Union[str, Path],
        sample_rate: int = 16000,
        max_length: Union[int, float] = 5,
        device: Optional[torch.device] = None,
        n_mels: int = 80,
        speakers: Optional[Iterable[int]] = None,
        augmentations: bool = False,
        split_name: str = "all",
    ):
        df = _load_pairs(pairs_file)
        if speakers is not None:
            keep = set(int(s) for s in speakers)
            df = df[df["speaker"].astype(int).isin(keep)].reset_index(drop=True)
        self.data = df

        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.hop_length = 160
        self.n_fft = 400
        self.augmentations = augmentations
        self.max_length = int(max_length * sample_rate)
        assert self.max_length % self.hop_length == 0, (
            "max_length must be multiple of hop_length"
        )
        self.n_mels = n_mels
        self.device = device or torch.device("cpu")

        phash = _file_hash(pairs_file)
        self.cache_file = (
            Path(__file__).parent
            / "assets"
            / f"paired_logmel_{Path(pairs_file).stem}_{n_mels}mels_{self.max_length}samples_{split_name}_{phash}.pt"
        )

        if not self.cache_file.exists():
            print(f"Precomputing paired LogMels ({len(self.data)} pairs)...")
            before_list, after_list, spk_list = [], [], []

            for idx in range(len(self.data)):
                row = self.data.iloc[idx]
                before_path = self.data_root / str(row["before"])
                after_path = self.data_root / str(row["after"])

                before_list.append(self._compute_logmel(str(before_path)))
                after_list.append(self._compute_logmel(str(after_path)))
                spk_list.append(int(row["speaker"]))

            self.before_specs = torch.stack(before_list).cpu()
            self.after_specs = torch.stack(after_list).cpu()
            self.speakers = torch.tensor(spk_list, dtype=torch.long)

            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "before": self.before_specs,
                    "after": self.after_specs,
                    "speakers": self.speakers,
                },
                self.cache_file,
            )
            print(f"Cached {len(self.speakers)} pairs → {self.cache_file}")
        else:
            print(f"Loading cached paired LogMels from {self.cache_file}")
            cache = torch.load(self.cache_file, weights_only=True)
            self.before_specs = cache["before"]
            self.after_specs = cache["after"]
            self.speakers = cache["speakers"]

    def _compute_logmel(self, audio_path: str) -> torch.Tensor:
        dec = AudioDecoder(audio_path, sample_rate=self.sample_rate, num_channels=1)
        signal = dec.get_all_samples().data  # [1, num_samples]
        signal = pad_or_trim(signal, self.max_length).to(self.device)

        window = torch.hann_window(self.n_fft, device=self.device)
        stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs() ** 2

        filters = mel_filters(self.device, self.n_mels)
        mel_spec = filters @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        spec_before = self.before_specs[idx]
        spec_after = self.after_specs[idx]
        if self.augmentations:
            spec_before = spec_augment(spec_before)
            spec_after = spec_augment(spec_after)
        return spec_before, spec_after


def make_paired_splits(
    pairs_file: str,
    data_root: Union[str, Path],
    train_speakers: Iterable[int],
    val_speakers: Iterable[int],
    test_speakers: Iterable[int],
    **kwargs,
):
    aug = kwargs.pop("augmentations", False)
    return (
        PairedLogMel_Dataset(
            pairs_file, data_root, speakers=train_speakers, split_name="train", augmentations=aug, **kwargs,
        ),
        PairedLogMel_Dataset(
            pairs_file, data_root, speakers=val_speakers, split_name="val", **kwargs,
        ),
        PairedLogMel_Dataset(
            pairs_file, data_root, speakers=test_speakers, split_name="test", **kwargs,
        ),
    )
