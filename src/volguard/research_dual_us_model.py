"""Optional NASDAQ and VIX paths, keeping the long daily core on every row."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(10,24,batch_first=True)
        self.head=nn.Linear(24,1)
        self.nasdaq=nn.Sequential(nn.Linear(30,8),nn.Tanh(),nn.Linear(8,1))
        self.vix=nn.Sequential(nn.Linear(30,8),nn.Tanh(),nn.Linear(8,1))
        for branch in [self.nasdaq,self.vix]:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    def forward(self,windows,nasdaq,nasdaq_mask,vix,vix_mask,use_nasdaq,use_vix):
        h=self.encoder(windows)[0][:,-1]
        score=self.head(h).squeeze(-1)
        for values,mask,enabled,branch in [(nasdaq,nasdaq_mask,use_nasdaq,self.nasdaq),(vix,vix_mask,use_vix,self.vix)]:
            if enabled and mask.any():
                if not torch.isfinite(values[mask]).all():
                    raise ValueError('Available context contains invalid values')
                score=score.clone()
                score[mask]=score[mask]+branch(torch.cat([h[mask],values[mask]],dim=1)).squeeze(-1)
        return score
