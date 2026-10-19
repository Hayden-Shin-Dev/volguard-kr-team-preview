"""Binary probability refinement on a time-OOF foundation logit."""
import numpy as np
from scipy.special import expit,logit


def offset_probability(model,features,base_probability):
    p=np.asarray(base_probability,dtype=float)
    if not np.isfinite(p).all() or not ((p>0)&(p<1)).all():
        raise ValueError('Foundation probabilities must be finite and strictly inside (0,1)')
    if not np.array_equal(model.classes_,[0,1]):raise ValueError('Unexpected binary class order')
    correction=np.asarray(model.predict(features,raw_score=True),dtype=float)
    if correction.shape!=p.shape or not np.isfinite(correction).all():raise ValueError('Invalid logit correction')
    return expit(logit(p)+correction)
