import torch.nn as nn


class CNN_Simple(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout),
            nn.MaxPool2d((2, 2)),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, feats):
        """
        feats: [B, n_mels, T]    — mfcc или patch
               [B, 1, n_mels, T] — logmel (channel убирается перед unsqueeze)
        """
        if feats.dim() == 4:
            feats = feats.squeeze(1)  # [B, 1, n_mels, T] → [B, n_mels, T]
        x = feats.unsqueeze(1)  # [B, 1, n_mels, T]
        x = self.cnn(x)  # [B, 64, T', F']
        x = self.global_pool(x)  # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        out = self.fc(x)  # [B]
        return out


class CNN(nn.Module):
    def __init__(self, n_mfcc, dropout=0.3):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Sequential(
            nn.Linear(64 * int(self.n_mfcc / 4), 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, feats):
        """
        feats: [B, n_mfcc, T]    — mfcc или patch
               [B, 1, n_mfcc, T] — logmel (channel убирается перед unsqueeze)
        """
        if feats.dim() == 4:
            feats = feats.squeeze(1)  # [B, 1, n_mfcc, T] → [B, n_mfcc, T]
        x = feats.unsqueeze(1)
        x = self.cnn(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

