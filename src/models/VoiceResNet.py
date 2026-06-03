import torch
import torch.nn as nn


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
        return x * noise.div_(survival)


class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, drop_path_prob: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        #self.se = SEBlock(out_channels)
        self.drop_path = DropPath(drop_path_prob)
        self.act = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.act(identity + self.drop_path((self.block(x))))  # + self.se(...) --- IGNORE ---


class VoiceResNet(nn.Module):

    def __init__(self, dropout: float = 0.4, drop_path_rate: float = 0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, 40, 250]
        )

        total_blocks = 6 
        dp = [drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)]

        self.layer1 = self._make_layer(32,  32,  blocks=2, stride=1, dp_rates=dp[0:2])
        self.layer2 = self._make_layer(32,  64,  blocks=2, stride=2, dp_rates=dp[2:4])
        self.layer3 = self._make_layer(64,  128, blocks=2, stride=2, dp_rates=dp[4:6])

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 2, 64),  # mean(128) + std(128)
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, ResBlock):
                last_bn = m.block[-1]
                assert isinstance(last_bn, nn.BatchNorm2d)
                nn.init.constant_(last_bn.weight, 0)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int, dp_rates: list[float]
    ) -> nn.Sequential:
        layers = [ResBlock(in_channels, out_channels, stride=stride, drop_path_prob=dp_rates[0])]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, drop_path_prob=dp_rates[i]))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        mean = x.mean(dim=[2, 3])
        std = x.std(dim=[2, 3], unbiased=False)
        return torch.cat([mean, std], dim=1)  # [B, 256]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))  # [B, 1]
