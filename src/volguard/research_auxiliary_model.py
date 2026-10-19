from torch import nn
from volguard.research_full_flow_model import Model as Base


class Model(Base):
    def __init__(self):
        super().__init__()
        self.auxiliary_head=nn.Linear(24,2)

    def forward(self,windows,context,mask,sequence):
        direction=super().forward(windows,context,mask,sequence)
        # Shared encoder, no future input; auxiliary head is never used at inference.
        hidden=self.encoder(windows)[0][:,-1]
        return direction,self.auxiliary_head(hidden)
