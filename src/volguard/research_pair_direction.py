"""Within-date discrimination auxiliary loss; never used by production."""
import torch
from torch.nn import functional as F


def paired_direction_loss(logits, labels, dates):
    """Average all opposite-label pairs within each date, then average dates.

    Training outcomes select pairs only inside the training loss. They must never
    determine input availability or the evaluation cohort.
    """
    terms = []
    for date in torch.unique(dates):
        selected = dates == date
        up = logits[selected & (labels == 1)]
        down = logits[selected & (labels == 0)]
        if up.numel() and down.numel():
            terms.append(F.softplus(down[None, :] - up[:, None]).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0
