"""Nested temporal residual learning, preserving the absolute predictor."""
import numpy as np
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from .oof_orthogonal import fit_detector as fit_absolute,COLUMNS
from .oof_relative import RANKS
from .oof_signal import assert_oof

def fit_detector(oof,kind):
    assert_oof(oof)
    absolute=fit_absolute(oof,'centered')
    if kind=='absolute':return (absolute,None)
    g=oof.dropna(subset=['absolute_oof']).copy()
    if not g.absolute_fit_last_target.lt(g.date).all():raise ValueError('Nested leakage')
    y=g.label-g.base_oof-g.absolute_oof;y=y-y.groupby(g.date).transform('mean')
    w=1/g.date.map(g.date.value_counts()).to_numpy();w/=w.mean()
    model=Ridge(alpha=1000) if kind=='linear' else LGBMRegressor(objective='regression',n_estimators=80,learning_rate=.02,num_leaves=3,min_child_samples=2000,reg_lambda=40,n_jobs=2,random_state=1721,verbosity=-1)
    model.fit(g[RANKS],y,sample_weight=w)
    return (absolute,model)

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    absolute,additional=detector
    correction=absolute.predict(frame[COLUMNS])
    if additional is not None:correction=correction+additional.predict(frame[RANKS])
    return np.clip(base+correction,1e-6,1-1e-6)
