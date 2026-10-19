"""Matched-capacity single-observation and masked-sequence option encoders."""
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.GRU(10,24,batch_first=True)
        self.head=nn.Linear(24,1)
        self.derivatives=nn.GRUCell(6,8)
        self.context=nn.Sequential(nn.Linear(32,8),nn.Tanh(),nn.Linear(8,1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self,windows,context,mask,sequence):
        daily=self.encoder(windows)[0][:,-1]
        logits=self.head(daily).squeeze(-1)
        # Identical availability for both modes; unavailable observations never enter the cell.
        eligible=mask[:,-1] & (mask.sum(dim=1)>=10)
        if not eligible.any():return logits
        state=context.new_zeros((len(context),8))  # neural initial state, not market imputation
        positions=range(context.shape[1]) if sequence else [context.shape[1]-1]
        for i in positions:
            valid=mask[:,i] & eligible
            if valid.any():
                updated=state.clone()
                updated[valid]=self.derivatives(context[valid,i],state[valid])
                state=updated
        result=logits.clone()
        result[eligible]+=self.context(torch.cat([daily[eligible],state[eligible]],dim=1)).squeeze(-1)
        return result
