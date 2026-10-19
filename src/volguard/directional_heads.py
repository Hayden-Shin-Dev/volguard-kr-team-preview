"""Monotone probability fusion, trained on an out-of-time labelled window."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit,expit

def inputs(up,down):
    return np.column_stack([logit(np.clip(up,1e-6,1-1e-6)),-logit(np.clip(down,1e-6,1-1e-6))])

def fit(up,down,y,weights):
    x=inputs(up,down);y=np.asarray(y);w=np.asarray(weights);w=w/w.sum()
    def loss(z):
        s=x@z[:2]+z[2]
        return float(w@(np.logaddexp(0,s)-y*s)+.001*np.sum(z[:2]**2))
    r=minimize(loss,[.5,.5,0],method='L-BFGS-B',bounds=[(0,10),(0,10),(-5,5)])
    if not r.success:raise ValueError('Head fusion did not converge')
    return r.x

def predict(z,up,down):
    return expit(inputs(up,down)@z[:2]+z[2])
