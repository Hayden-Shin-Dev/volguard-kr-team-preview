"""Research-only joint overnight/intraday return distribution."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def unpack(parameters):
    weights=torch.softmax(parameters[:,:,0],dim=1)
    means=parameters[:,:,1:3]
    scales=F.softplus(parameters[:,:,3:5])+.05
    correlation=.95*torch.tanh(parameters[:,:,5])
    return weights,means,scales,correlation

def up_probability(parameters):
    weights,means,scales,rho=unpack(parameters)
    total_mean=means.sum(dim=-1)
    total_variance=(scales**2).sum(dim=-1)+2*rho*scales.prod(dim=-1)
    # The sum of two jointly normal component returns is normal; mixture
    # probabilities are mixed with learned component weights, not multiplied.
    return (weights*torch.special.ndtr(total_mean/torch.sqrt(total_variance))).sum(dim=1)

def joint_nll(parameters,target):
    weights,means,scales,rho=unpack(parameters)
    residual=(target[:,None,:]-means)/scales
    one_minus=1-rho.square()
    exponent=(residual.square().sum(dim=-1)-2*rho*residual.prod(dim=-1))/one_minus
    log_density=-math.log(2*math.pi)-torch.log(scales).sum(dim=-1)-.5*torch.log(one_minus)-.5*exponent
    return -torch.logsumexp(torch.log(weights.clamp_min(1e-12))+log_density,dim=1)

class JointReturnGRU(nn.Module):
    def __init__(self,features,components=3):
        super().__init__();self.components=components;self.encoder=nn.GRU(features,24,batch_first=True);self.head=nn.Linear(24,components*6)
    def forward(self,x):return self.head(self.encoder(x)[0][:,-1]).reshape(-1,self.components,6)
