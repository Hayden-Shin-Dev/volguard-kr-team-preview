"""Paired GRU and order-invariant daily-window representation; research only."""
import torch
from torch import nn
from .research_stock_effect_model import Model as Base

class Model(Base):
    def __init__(self, stock_count):
        super().__init__(stock_count)
        self.local=nn.Sequential(nn.Linear(10,24),nn.Tanh())
        self.pooled_head=nn.Sequential(nn.Linear(48,24),nn.Tanh(),nn.Linear(24,1))

    def forward(self,x,stock_ids,mode):
        if mode=='shared':return super().forward(x,stock_ids,'shared')
        if mode!='set_pool':raise ValueError('Unknown representation')
        local=self.local(x)
        return self.pooled_head(torch.cat([local.mean(1),local[:,-1]],dim=1)).squeeze(-1)
