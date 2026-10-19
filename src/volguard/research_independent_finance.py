"""Separate observed-source encoders; missing sources never disable other sources."""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, sizes=(4, 14, 4)):
        super().__init__()
        self.sizes = sizes
        self.encoder = nn.GRU(10, 24, batch_first=True)
        self.head = nn.Linear(24, 1)
        self.branches = nn.ModuleList(nn.GRUCell(n, 8) for n in sizes)
        self.context = nn.Sequential(nn.Linear(24 + 8 * len(sizes), 8), nn.Tanh(), nn.Linear(8, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, windows, context, mask, sequence=False):
        daily = self.encoder(windows)[0][:, -1]
        logits = self.head(daily).squeeze(-1)
        eligible = mask[:, -1] & (mask.sum(dim=1) >= 10)
        states = []
        offset = 0
        for group, (size, cell) in enumerate(zip(self.sizes, self.branches)):
            state = context.new_zeros((len(context), 8))
            for i in (range(context.shape[1]) if sequence else [context.shape[1] - 1]):
                valid = mask[:, i, group] & eligible[:, group]
                if valid.any():
                    updated = state.clone()
                    updated[valid] = cell(context[valid, i, offset:offset + size], state[valid])
                    state = updated
            states.append(state)
            offset += size
        active = eligible.any(dim=1)
        if active.any():
            logits = logits.clone()
            logits[active] += self.context(torch.cat([daily] + states, dim=1)[active]).squeeze(-1)
        return logits
