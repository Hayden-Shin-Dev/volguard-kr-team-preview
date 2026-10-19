"""Same parameter budget as recurrent context; valid historical mean then latest."""
import torch
from volguard.research_derivatives_sequence import Model as SequenceModel


class Model(SequenceModel):
    def forward(self, windows, context, mask, sequence=True):
        daily = self.encoder(windows)[0][:, -1]
        logits = self.head(daily).squeeze(-1)
        eligible = mask[:, -1] & (mask.sum(dim=1) >= 10)
        if not eligible.any():
            return logits
        values = context[eligible]
        valid = mask[eligible]
        # Zero is only the additive identity in a masked sum; missing market
        # values never enter either recurrent-cell observation.
        mean = torch.where(valid[:, :, None], values, 0).sum(1) / valid.sum(1)[:, None]
        state = self.derivatives(mean, mean.new_zeros((len(mean), 8)))
        state = self.derivatives(values[:, -1], state)
        result = logits.clone()
        result[eligible] += self.context(torch.cat([daily[eligible], state], dim=1)).squeeze(-1)
        return result
