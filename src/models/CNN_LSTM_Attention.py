import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, lstm_hidden=64, lstm_layers=1, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d((1,2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d((2,2))
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1)) 

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, feats):
        """
        feats: [B, T, F] — спектрограмма / MFCC
        """
        x = feats.unsqueeze(1)          # [B, 1, T, F]
        x = self.cnn(x)                 # [B, 64, T', F']
        x = self.adaptive_pool(x)       # [B, 64, T', 1]
        x = x.squeeze(-1)               # [B, 64, T']
        x = x.permute(0, 2, 1)          # [B, T', 64] для LSTM

        lstm_out, _ = self.lstm(x)      # [B, T', 2*lstm_hidden]

        attn_scores = self.attention(lstm_out)          # [B, T', 1]
        attn_weights = F.softmax(attn_scores, dim=1)    # нормируем по времени
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, 2*lstm_hidden]

        out = self.fc(context)          # [B, 1]
        return out
