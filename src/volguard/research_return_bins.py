"""Direct categorical return likelihood; sign probability sums disjoint bins."""
import numpy as np
import torch
from torch import nn

EDGES=np.array([-1.,-.25,0.,.25,1.])
def labels(returns,past_volatility):
    r=np.asarray(returns);v=np.asarray(past_volatility)
    if not np.isfinite(r).all() or not np.isfinite(v).all() or (v<=0).any() or (r==0).any():
        raise ValueError('Finite nonflat target and positive past volatility required')
    return np.searchsorted(EDGES,r/v,side='right')

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.GRU(10,24,batch_first=True);self.binary=nn.Linear(24,1);self.bins=nn.Linear(24,6)
    def forward(self,x,mode):
        h=self.encoder(x)[0][:,-1]
        if mode=='binary':return self.binary(h).squeeze(-1)
        if mode=='categorical':return self.bins(h)
        raise ValueError('Unknown objective')

def direction_logit(logits):
    return torch.logsumexp(logits[:,3:],dim=1)-torch.logsumexp(logits[:,:3],dim=1)
