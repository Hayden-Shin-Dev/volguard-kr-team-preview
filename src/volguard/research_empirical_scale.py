"""Frozen training-only marginal normalization, preserving missing observations."""
import numpy as np
from scipy.special import ndtri


def fit_grid(values):
    values=np.asarray(values,float)
    finite=values[np.isfinite(values)]
    if len(finite)<100:raise ValueError('Insufficient finite training observations')
    probabilities=np.linspace(.001,.999,999)
    quantiles=np.quantile(finite,probabilities)
    # Equal observed values must receive the same interpolated score.
    unique,inverse=np.unique(quantiles,return_inverse=True)
    scores=np.bincount(inverse,weights=probabilities)/np.bincount(inverse)
    return {'values':unique.tolist(),'probabilities':scores.tolist()}


def transform(values,grid):
    values=np.asarray(values,float)
    result=np.full(values.shape,np.nan)
    valid=np.isfinite(values)
    p=np.interp(values[valid],grid['values'],grid['probabilities'])
    result[valid]=ndtri(np.clip(p,.001,.999))
    return result
