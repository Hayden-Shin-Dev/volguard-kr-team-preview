"""20-session daily encoder with optional 60-session weekly-summary context."""
import torch
from torch import nn
from .research_stock_effect_model import Model as Base

class Model(Base):
    def __init__(self,stock_count):
        super().__init__(stock_count)
        self.long_encoder=nn.GRU(10,12,batch_first=True)
        self.long_head=nn.Linear(36,1)
        nn.init.zeros_(self.long_head.weight);nn.init.zeros_(self.long_head.bias)
    def forward(self,short,long,mask,mode):
        if mode not in ['short','multiscale']:raise ValueError('Unknown context mode')
        h=self.encoder(short)[0][:,-1];score=self.head(h).squeeze(-1)
        if mode=='multiscale' and mask.any():
            if long is None or len(long)!=int(mask.sum()):raise ValueError('Observed long windows required')
            z=self.long_encoder(long)[0][:,-1]
            score=score.clone();score[mask]+=self.long_head(torch.cat([h[mask],z],1)).squeeze(-1)
        return score
