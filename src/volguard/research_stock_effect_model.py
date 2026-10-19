"""Research daily core with optional train-only stock effects, not probability inflation."""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, stock_count):
        super().__init__()
        self.encoder=nn.GRU(10,24,batch_first=True)
        self.head=nn.Linear(24,1)
        self.bias=nn.Embedding(stock_count+1,1,padding_idx=0)
        self.effect=nn.Embedding(stock_count+1,4,padding_idx=0)
        self.project=nn.Linear(24,4,bias=False)
        nn.init.zeros_(self.bias.weight)
        nn.init.zeros_(self.effect.weight)

    def forward(self,x,stock_ids,mode):
        if mode not in ['shared','bias','response']:raise ValueError('Unknown stock effect mode')
        h=self.encoder(x)[0][:,-1]
        score=self.head(h).squeeze(-1)
        if mode!='shared':score=score+self.bias(stock_ids).squeeze(-1)
        if mode=='response':score=score+(self.effect(stock_ids)*self.project(h)).sum(dim=1)/2.
        return score
