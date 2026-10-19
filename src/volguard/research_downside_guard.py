import numpy as np


def compose(baseline,candidate):
    baseline=np.asarray(baseline,dtype=float);candidate=np.asarray(candidate,dtype=float)
    if baseline.shape!=candidate.shape or baseline.ndim!=1:
        raise ValueError('Aligned one-dimensional probabilities required')
    for p in [baseline,candidate]:
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probability')
    return np.where(baseline<=.4,baseline,candidate)
