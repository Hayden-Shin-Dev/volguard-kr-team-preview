"""Research-only numeric adapter; never mutates deployed calibrators."""
import numpy as np
from .long_history_probability import positive_calibration

def fit_positive(p,y):
    probabilities=np.asarray(p,dtype=np.float64)
    labels=np.asarray(y,dtype=np.float64)
    if probabilities.ndim!=1 or labels.shape!=probabilities.shape:
        raise ValueError('One-dimensional aligned probabilities and labels required')
    if not np.isfinite(probabilities).all() or not np.isin(labels,[0,1]).all():
        raise ValueError('Finite probabilities and binary labels required')
    if not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError('Probability outside [0,1]')
    return positive_calibration(probabilities,labels)
