from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.data.paired_dataset import make_paired_splits
from src.models.GAN import UNet, PatchGAN
from src.models.VoiceResNet import VoiceResNet
from src.visualizations.viz_utils import (
    plot_gan_history,
    save_audio_triplet,
    save_spectrogram_tensor,
    save_spectrogram_triplet,
)


class Config:
    # --- Модель ---
    model = "pixel2pixel"
    discriminator: str = "resnet"  # "patchgan" или "resnet"
    # --- Обучение ---
    checkpoint_freq = 5
    lr: float = 2e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 70
    max_patience: int = 30
    lambda_l1: float = 10   # вес L1 в pix2pix-лоссе генератора

    # --- Датасет ---
    n_mels: int = 128
    max_length: float = (
        5.12  # секунды → 81920 сэмплов → 512 фреймов при center=True STFT (кратно 64)
    )
    augmentations: bool = True

    # Paired pix2pix датасет: таблица (before, after, speaker)
    pairs_file: str = "/workspaces/super-duper-dollop/data/pair_dataset.ods"
    data_root: str = "/workspaces/super-duper-dollop"
    # Speaker-disjoint сплит (speakers 1..11 в pair_dataset.ods)
    train_speakers: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    val_speakers: tuple = (17, 18)
    test_speakers: tuple = (19, 20)

    # --- Воспроизводимость ---
    seed: int = 555

    # --- Визуализация ---
    viz_dir: str = "/workspaces/super-duper-dollop/gan_viz"
    viz_every: int = 5  # сохранять примеры раз в N эпох


def _unpack_pair(batch, device):
    """Paired-dataset returned (spec_before, spec_after).
    real_a — до операции, real_b — после"""
    real_a, real_b = batch
    real_a = real_a.to(device, dtype=torch.float32)
    real_b = real_b.to(device, dtype=torch.float32)
    if real_a.dim() == 3:
        real_a = real_a.unsqueeze(1)
    if real_b.dim() == 3:
        real_b = real_b.unsqueeze(1)
    return real_a, real_b


def build_discriminator(name: str, device):
    if name == "patchgan":
        return PatchGAN(in_channels=1).to(device)
    elif name == "resnet":
        return VoiceResNet(dropout=0.4, drop_path_rate=0.1).to(device)
    else:
        raise ValueError(f"Unknown discriminator: {name!r}")


def evaluate_gan(G, D, loader, crit_adv, crit_l1, lambda_l1, device):
    G.eval()
    D.eval()
    total_G, total_D, total_L1, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            real_a, real_b = _unpack_pair(batch, device)
            fake_a = G(real_b)
            # fake_AB = torch.cat((real_b, fake_a), 1)
            pred_fake = D(fake_a)
            real_labels = torch.ones_like(pred_fake)
            fake_labels = torch.zeros_like(pred_fake)

            l1 = crit_l1(fake_a, real_a)
            loss_G = crit_adv(pred_fake, real_labels) + lambda_l1 * l1
            # real_AB = torch.cat((real_b, real_a), 1)
            pred_real = D(real_a)
            loss_D = 0.5 * (
                crit_adv(pred_real, real_labels) + crit_adv(pred_fake, fake_labels)
            )

            total_G += loss_G.item()
            total_D += loss_D.item()
            total_L1 += l1.item()
            n += 1
    return total_G / max(n, 1), total_D / max(n, 1), total_L1 / max(n, 1)


def train(device, cfg: Config, train_loader, val_loader):
    # real_a - до операции
    # real_b - после операции
    G = UNet(in_channels=1, out_channels=1).to(device)
    D = build_discriminator(cfg.discriminator, device)

    optim_G = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optim_D = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    crit_adv = nn.MSELoss()
    crit_l1 = nn.L1Loss()

    viz_dir = Path(cfg.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    best_val_l1 = float("inf")
    patience_counter = 0
    ckpt_path = f"best_{cfg.model}.pt"

    history = {
        "train_G": [],
        "train_D": [],
        "val_G": [],
        "val_D": [],
        "val_L1": [],
    }

    epoch_width = len(str(cfg.max_epochs))

    for e in range(1, cfg.max_epochs + 1):
        G.train()
        D.train()
        run_G, run_D, n_batches = 0.0, 0.0, 0

        loop = tqdm(
            train_loader, desc=f"Epoch {e:>{epoch_width}}/{cfg.max_epochs}", leave=False
        )
        for batch in loop:
            real_a, real_b = _unpack_pair(batch, device)

            # --- Train generator ---
            optim_G.zero_grad()
            fake_a = G(real_b)
            # CGAN
            # fake_AB = torch.cat((real_b, fake_a), 1)

            pred_fake = D(fake_a)
            real_labels = torch.ones_like(pred_fake)
            fake_labels = torch.zeros_like(pred_fake)

            loss_G_adv = crit_adv(pred_fake, real_labels)
            loss_G_l1 = crit_l1(fake_a, real_a) * cfg.lambda_l1
            loss_G = loss_G_adv + loss_G_l1
            loss_G.backward()
            optim_G.step()

            # --- Train discriminator ---
            # Fake
            # fake_AB = torch.cat((real_b, fake_a), 1).detach()
            # Real
            # real_AB = torch.cat((real_b, real_a), 1)
            optim_D.zero_grad()
            pred_real = D(real_a)
            loss_real = crit_adv(pred_real, real_labels)
            pred_fake = D(fake_a.detach())
            loss_fake = crit_adv(pred_fake, fake_labels)

            loss_D = 0.5 * (loss_fake + loss_real)
            loss_D.backward()
            optim_D.step()

            run_G += loss_G.item()
            run_D += loss_D.item()
            n_batches += 1
            loop.set_postfix(G=f"{loss_G.item():.3f}", D=f"{loss_D.item():.3f}")

        train_G = run_G / max(n_batches, 1)
        train_D = run_D / max(n_batches, 1)
        val_G, val_D, val_L1 = evaluate_gan(
            G, D, val_loader, crit_adv, crit_l1, cfg.lambda_l1, device
        )

        history["train_G"].append(train_G)
        history["train_D"].append(train_D)
        history["val_G"].append(val_G)
        history["val_D"].append(val_D)
        history["val_L1"].append(val_L1)

        print(
            f"[Epoch {e:>{epoch_width}}/{cfg.max_epochs}] "
            f"train G={train_G:.4f} D={train_D:.4f} | "
            f"val G={val_G:.4f} D={val_D:.4f} L1={val_L1:.4f}"
        )

        if e % cfg.viz_every == 0:
            G.eval()
            with torch.no_grad():
                real_a, real_b = _unpack_pair(next(iter(val_loader)), device)
                fake_a = G(real_b)
            save_spectrogram_triplet(
                real_b[0].cpu(),  # source: после операции
                real_a[0].cpu(),  # target: до операции
                fake_a[0].cpu(),  # generated: до операции
                viz_dir / f"epoch_{e:0{epoch_width}}.png",
                title=f"Epoch {e}",
            )
            G.train()

        if val_L1 < best_val_l1:
            best_val_l1 = val_L1
            patience_counter = 0
            torch.save(
                {
                    "epoch": e,
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "val_G_loss": val_G,
                    "val_D_loss": val_D,
                    "val_L1": val_L1,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.max_patience:
                print(f"Early stopping at epoch {e} | best val L1={best_val_l1:.4f}")
                break

        if e % cfg.checkpoint_freq == 0:
            torch.save(
                {
                    "epoch": e,
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "G_loss": train_G,
                    "D_loss": train_D,
                },
                Path(ckpt_path).as_posix(),
            )
            G.eval()
            with torch.no_grad():
                sample_a, sample_b = _unpack_pair(next(iter(val_loader)), device)
                sample_fake = G(sample_b)
                save_audio_triplet(
                    sample_b[0].cpu(),
                    sample_a[0].cpu(),
                    sample_fake[0].cpu(),
                    viz_dir / f"audio_epoch_{e:0{epoch_width}}",
                    n_mels=cfg.n_mels,
                )
                save_spectrogram_tensor(
                    sample_fake[0].cpu(), viz_dir / f"epoch_{e:0{epoch_width}}_fake"
                )
                save_spectrogram_tensor(
                    sample_a[0].cpu(), viz_dir / f"epoch_{e:0{epoch_width}}_real_a"
                )
                save_spectrogram_tensor(
                    sample_b[0].cpu(), viz_dir / f"epoch_{e:0{epoch_width}}_real_b"
                )
            G.train()

    plot_gan_history(history, "gan_training_plots.png")
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = Config()

    train_ds, val_ds, test_ds = make_paired_splits(
        cfg.pairs_file,
        cfg.data_root,
        train_speakers=cfg.train_speakers,
        val_speakers=cfg.val_speakers,
        test_speakers=cfg.test_speakers,
        n_mels=cfg.n_mels,
        max_length=cfg.max_length,
        device=device,
        augmentations=cfg.augmentations,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    train(device, cfg, train_loader, val_loader)


if __name__ == "__main__":
    main()
