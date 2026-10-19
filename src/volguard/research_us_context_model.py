"""Frozen architecture for optional archived US context; research only."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(10,24,batch_first=True)
        self.head=nn.Linear(24,1)
        self.context=nn.Sequential(nn.Linear(30,8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self,windows,u,mask,enabled):
        h=self.encoder(windows)[0][:,-1]
        p=self.head(h).squeeze(-1)
        if enabled and mask.any():
            p=p.clone()
            p[mask]=p[mask]+self.context(torch.cat([h[mask],u[mask]],dim=1)).squeeze(-1)
        return p
