"""Daily direction core with an optional, explicitly masked futures residual."""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, context_indices=(0, 1, 2, 3)):
        super().__init__()
        self.encoder = nn.GRU(10, 24, batch_first=True)
        self.head = nn.Linear(24, 1)
        self.context_indices = tuple(context_indices)
        self.context = nn.Sequential(nn.Linear(24 + len(self.context_indices), 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, windows, context, mask, enabled):
        h = self.encoder(windows)[0][:, -1]
        logits = self.head(h).squeeze(-1)
        if enabled and mask.any():
            logits = logits.clone()
            logits[mask] = logits[mask] + self.context(torch.cat([h[mask], context[mask][:, self.context_indices]], dim=1)).squeeze(-1)
        return logits
