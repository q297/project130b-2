import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, normalization=True, stride=2, track_stats=False
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not normalization,
            )
        )
        if normalization:
            self.layers.append(
                nn.InstanceNorm2d(out_channels, track_running_stats=track_stats)
            )
        self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, track_stats=False):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )

        self.layers.append(
            nn.InstanceNorm2d(out_channels, track_running_stats=track_stats)
        )

        if dropout:
            self.layers.append(nn.Dropout2d(0.5))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(in_channels, 64, normalization=False),
                EncoderBlock(64, 128),
                EncoderBlock(128, 256),
                EncoderBlock(256, 512),
                EncoderBlock(512, 512),
                EncoderBlock(512, 512),
                EncoderBlock(512, 512),

            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(512, 512, dropout=True),
                DecoderBlock(1024, 512, dropout=True),
                DecoderBlock(1024, 512),
                DecoderBlock(1024, 256),
                DecoderBlock(512, 128),
                DecoderBlock(256, 64),
            ]
        )
        self.last_conv = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.out_activation = nn.Tanh()

    def forward(self, x):
        skips = []
        for l in self.encoder:
            x = l(x)
            skips.insert(0, x)
        for s, l in zip(skips[1:], self.decoder):
            x = l(x)
            x = torch.cat((s, x), dim=1)
        return self.out_activation(self.last_conv(x))


class PatchGAN(nn.Module):
    def __init__(self, in_channels, sigmoid=False):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(in_channels, 64, normalization=False),
                EncoderBlock(64, 128),
                EncoderBlock(128, 256),
                EncoderBlock(256, 512, stride=1),
                nn.Conv2d(
                    in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1
                ),
            ]
        )
        if sigmoid:
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
