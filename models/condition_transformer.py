"""Pooled or shared-encoder conditional Transformer. No current input at inference."""

import copy
import math
import torch
from torch import nn


class ConditionTransformer(nn.Module):
    def __init__(self, heads=1, width=128, layers=3, dropout=0.1):
        super().__init__()
        self.input = nn.Linear(5, width)
        pos = torch.arange(200)[:, None]
        rate = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(200, width)
        pe[:, 0::2] = torch.sin(pos * rate)
        pe[:, 1::2] = torch.cos(pos * rate)
        self.register_buffer("position", pe[None])
        layer = nn.TransformerEncoderLayer(
            width, 4, width * 2, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )
        self.heads = nn.ModuleList([copy.deepcopy(head) for _ in range(heads)])

    def forward(self, x):
        z = self.encoder(self.input(x) + self.position[:, : x.shape[1]]).mean(1)
        return torch.cat([head(z) for head in self.heads], dim=1)
