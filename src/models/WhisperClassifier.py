
import numpy as np
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> Tensor:
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.attn_dropout = dropout
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(self, x: Tensor) -> Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        n_batch, n_ctx, n_state = q.shape
        head_dim = n_state // self.n_head

        # reshape → [B, n_head, T, head_dim]
        q = q.view(n_batch, n_ctx, self.n_head, head_dim).permute(0, 2, 1, 3)
        k = k.view(n_batch, n_ctx, self.n_head, head_dim).permute(0, 2, 1, 3)
        v = v.view(n_batch, n_ctx, self.n_head, head_dim).permute(0, 2, 1, 3)

        # dropout_p применяется только во время train(); при eval() PyTorch игнорирует его
        dp = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)

        out = out.permute(0, 2, 1, 3).flatten(start_dim=2)  # [B, T, n_state]
        return self.out(out)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, dropout)
        self.attn_ln = LayerNorm(n_state)
        self.attn_drop = nn.Dropout(dropout)  # после attention, перед residual add

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(n_mlp, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)
        self.mlp_drop = nn.Dropout(dropout)   # после MLP, перед residual add

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_drop(self.attn(self.attn_ln(x)))
        x = x + self.mlp_drop(self.mlp(self.mlp_ln(x)))
        return x


class AudioEncoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor) -> Tensor:
        """
        x:       [B, n_mels, T]
        returns: [B, T//2, n_state]
        """
        x = F.gelu(self.conv1(x))  # [B, n_state, T]
        x = F.gelu(self.conv2(x))  # [B, n_state, T//2]
        x = x.permute(0, 2, 1)  # [B, T//2, n_state]

        T = x.shape[1]
        x = (x + self.positional_embedding[:T]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)  # [B, T//2, n_state]
        return x

class WhisperClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        n_ctx: int = 250,  # T//2 для max_length=5 сек (T≈500 → 250 после stride-2)
        n_state: int = 128,  # размерность скрытого состояния
        n_head: int = 4,
        n_layer: int = 4,  # число Transformer-блоков
        dropout: float = 0.3,
    ):
        super().__init__()
        assert n_state % n_head == 0, (
            f"n_state ({n_state}) must be divisible by n_head ({n_head})"
        )

        self.encoder = AudioEncoder(
            n_mels=n_mels,
            n_ctx=n_ctx,
            n_state=n_state,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_state * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [B, 1, n_mels, T]  — из LogMel_Dataset (batched)
        или [B, n_mels, T]    — из PatchLogMel_Dataset
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, n_mels, T]

        x = self.encoder(x)  # [B, T//2, n_state]

        mean = x.mean(dim=1)  # [B, n_state]
        std = x.std(dim=1, unbiased=False)  # [B, n_state]
        x = torch.cat([mean, std], dim=1)  # [B, n_state * 2]

        return self.head(x)  # [B, 1]
