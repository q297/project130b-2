import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
EPOCHS = 500
BATCH_SIZE = 16
LR = 1e-3
SEGMENT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEL_MEAN = -4.0
MEL_STD = 4.0

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
).to(DEVICE)

_inverse_mel = T.InverseMelScale(
    n_stft=N_FFT // 2 + 1,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE,
).cpu()

_griffin_lim = T.GriffinLim(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_iter=64,
).cpu()


def wav_to_mel(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(dim=0, keepdim=True).to(DEVICE)
    mel = _mel_transform(wav).squeeze(0)
    mel = torch.log(mel + 1e-9)
    mel = (mel - MEL_MEAN) / MEL_STD
    return mel


def mel_to_wav(mel: torch.Tensor) -> torch.Tensor:
    mel = mel * MEL_STD + MEL_MEAN
    mel = torch.exp(mel).cpu()
    spec = _inverse_mel(mel)
    wav = _griffin_lim(spec)
    return wav.unsqueeze(0)


class VoiceDataset(Dataset):
    def __init__(self, before_dir: str, after_dir: str, segment_len: int = SEGMENT_LEN):
        self.segment_len = segment_len
        before_files = sorted(Path(before_dir).glob("*.wav"))
        after_files = sorted(Path(after_dir).glob("*.wav"))

        before_names = {f.stem: f for f in before_files}
        after_names = {f.stem: f for f in after_files}
        common = sorted(set(before_names) & set(after_names))
        assert len(common) > 0, "No matching file pairs found!"

        self.pairs = [(before_names[n], after_names[n]) for n in common]
        print(f"Pairs found: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        before_path, after_path = self.pairs[idx]
        before_mel = wav_to_mel(str(before_path))
        after_mel = wav_to_mel(str(after_path))

        min_len = min(before_mel.shape[1], after_mel.shape[1])
        if min_len < self.segment_len:
            pad = self.segment_len - min_len
            before_mel = F.pad(before_mel, (0, pad))
            after_mel = F.pad(after_mel, (0, pad))
            start = 0
        else:
            start = torch.randint(0, min_len - self.segment_len + 1, (1,)).item()

        return (
            after_mel[:, start : start + self.segment_len],
            before_mel[:, start : start + self.segment_len],
        )


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 5, stride=2, padding=2),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm1d(out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class VoiceConverter(nn.Module):
    def __init__(self, n_mels: int = N_MELS):
        super().__init__()
        self.in_conv = nn.Conv1d(n_mels, 128, 3, padding=1)

        self.down1 = DownBlock(128, 256)
        self.down2 = DownBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=2, dilation=2),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 3, padding=4, dilation=4),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2),
        )

        self.up2 = UpBlock(512, 256)
        self.up1 = UpBlock(512, 128)

        self.out_conv = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, n_mels, 1),
        )

    def forward(self, x):
        x0 = self.in_conv(x)

        d1 = self.down1(x0)
        d2 = self.down2(d1)

        b = self.bottleneck(d2)

        u2 = self.up2(b)
        if u2.shape[-1] != d1.shape[-1]:
            u2 = F.interpolate(u2, size=d1.shape[-1])
        u2 = torch.cat([u2, d1], dim=1)

        u1 = self.up1(u2)
        if u1.shape[-1] != x0.shape[-1]:
            u1 = F.interpolate(u1, size=x0.shape[-1])
        u1 = torch.cat([u1, x0], dim=1)

        return self.out_conv(u1) + x


def train(before_dir: str, after_dir: str, model_path: str, epochs: int = EPOCHS):
    print(f"Device: {DEVICE}")

    dataset = VoiceDataset(before_dir, after_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VoiceConverter().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.L1Loss()
    scaler = GradScaler("cuda")

    best_loss = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for after_mel, before_mel in loader:
            after_mel = after_mel.to(DEVICE)
            before_mel = before_mel.to(DEVICE)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(after_mel)
                loss = criterion(pred, before_mel)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

    print(f"Done. Best loss: {best_loss:.4f} | Saved: {model_path}")


def train_all_speakers(dataset_root: str, models_dir: str, epochs: int = EPOCHS):
    root = Path(dataset_root)
    speakers = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"Found {len(speakers)} speakers")

    for spk in speakers:
        before_dir = spk / "before"
        after_dir = spk / "after"
        if not before_dir.exists() or not after_dir.exists():
            print(f"Skipping {spk.name}: missing before/after folders")
            continue

        model_path = Path(models_dir) / f"{spk.name}.pt"
        if model_path.exists():
            print(f"Skipping {spk.name}: model already exists")
            continue

        print(f"\n=== Training {spk.name} ===")
        train(str(before_dir), str(after_dir), str(model_path), epochs)


def convert(input_path: str, model_path: str, output_path: str):
    model = VoiceConverter().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    mel = wav_to_mel(input_path).unsqueeze(0)

    with torch.no_grad():
        converted_mel = model(mel).squeeze(0)

    wav = mel_to_wav(converted_mel)
    torchaudio.save(output_path, wav, SAMPLE_RATE)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--before", required=True)
    t.add_argument("--after", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--epochs", type=int, default=EPOCHS)

    ta = sub.add_parser("train-all")
    ta.add_argument("--dataset", required=True, help="Root dir with speaker_XX folders")
    ta.add_argument("--models", required=True, help="Dir to save per-speaker models")
    ta.add_argument("--epochs", type=int, default=EPOCHS)

    c = sub.add_parser("convert")
    c.add_argument("--input", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        train(args.before, args.after, args.model, args.epochs)
    elif args.cmd == "train-all":
        train_all_speakers(args.dataset, args.models, args.epochs)
    elif args.cmd == "convert":
        convert(args.input, args.model, args.output)
    else:
        parser.print_help()
