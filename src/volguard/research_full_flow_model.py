"""Daily core retained for every row; masked flow branch only for observed histories."""
from torch import nn
from volguard.research_derivatives_sequence import Model as MaskedContextModel


class Model(MaskedContextModel):
    def __init__(self):
        super().__init__()
        self.derivatives = nn.GRUCell(14, 8)
