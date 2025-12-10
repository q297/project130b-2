import pandas as pd
import torch
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder
import torchaudio
from torch_audiomentations import (
    Compose,
    Identity,
    AddColoredNoise,
    Shift,
    Gain,
    PolarityInversion,
    PeakNormalization,
    PitchShift,
)


class SpeakerDataset(Dataset):
    def __init__(
        self,
        parquet_file: str,
        sample_rate: int = 16000,
        max_length: int = 3,
        n_mfcc: int = 32,
        augmentations: bool = False,
    ):
        """
        Speaker dataset with TorchCodec decoding.

        Args:
            parquet_file (str): Path to the parquet file containing audio file paths and labels.
            sample_rate (int, optional): Desired sample rate for audio files. Defaults to 16000.
            max_length (int, optional): Max audio length in seconds. Defaults to 5.
            n_mfcc (int, optional): Number of MFCC. Defaults to 52.
            augmentations (bool, optional): Whether to apply data augmentations. Defaults to False.
        """
        self.data = (
            pd.read_parquet(parquet_file).dropna().reset_index(drop=True).drop(0)
        )
        self.sample_rate = sample_rate
        self.max_length = int(max_length * sample_rate)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        )
        self.augmentations = (
            Compose(
                transforms=[
                    Gain(
                        min_gain_in_db=-6, max_gain_in_db=6, p=0.5, output_type="dict"
                    ),
                    AddColoredNoise(
                        min_snr_in_db=20, max_snr_in_db=35, p=0.5, output_type="dict"
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
                    # PitchShift(
                    #     min_transpose_semitones=-1,
                    #     max_transpose_semitones=1,
                    #     p=0.2,
                    #     sample_rate=sample_rate,
                    #     output_type="dict",
                    # ),
                ],
                output_type="dict", shuffle=True
            )
            if augmentations
            else Identity(output_type="dict")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.data.iloc[idx, 0]
        label = float(self.data.iloc[idx, 1])
        dec = AudioDecoder(audio_path, sample_rate=self.sample_rate, num_channels=1)
        samples = dec.get_all_samples()
        signal = samples.data  # [1, num_samples]

        signal = signal.unsqueeze(0)  # [B=1, C=1, T=num_samples]
        signal = self.augmentations(signal, sample_rate=self.sample_rate).samples
        signal = signal.squeeze(0)  # [C, T] для дальнейшей обработки

        # 🔹 Pad / crop to fixed length
        length = signal.shape[1]
        if length > self.max_length:
            start = torch.randint(0, length - self.max_length + 1, (1,)).item()
            signal = signal[:, start : start + self.max_length]
        else:
            pad = self.max_length - length
            signal = torch.cat([signal, torch.zeros(signal.shape[0], pad)], dim=1)
        # 🔹 Extract MFCC
        mfcc = self.mfcc_transform(signal).squeeze(0)

        # 🔹 Normalize MFCC
        mean = mfcc.mean(dim=1, keepdim=True)
        std = mfcc.std(dim=1, keepdim=True) + 1e-9
        mfcc_norm = (mfcc - mean) / std

        return mfcc_norm, label
