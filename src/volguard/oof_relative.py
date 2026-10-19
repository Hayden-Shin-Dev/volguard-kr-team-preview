"""Observable cross-sectional ranks for date-centered OOF residual research."""
import numpy as np
from lightgbm import LGBMRegressor
from .oof_orthogonal import COLUMNS, residual_target
from .oof_signal import assert_oof

RANKS=['rank_'+c for c in COLUMNS]

def add_ranks(all_observed):
    result=all_observed.copy()
    for col, rank in zip(COLUMNS,RANKS):
        result[rank]=result.groupby('date')[col].rank(pct=True)-.5
    return result

def columns(kind):
    return COLUMNS if kind=='absolute' else RANKS if kind=='relative' else COLUMNS+RANKS

def fit_detector(oof,kind):
    assert_oof(oof)
    weights=1/oof.date.map(oof.date.value_counts()).to_numpy()
    weights/=weights.mean()
    model=LGBMRegressor(objective='regression',n_estimators=120,learning_rate=.025,num_leaves=7,min_child_samples=1000,reg_lambda=20,n_jobs=2,random_state=1721,verbosity=-1)
    model.fit(oof[columns(kind)],residual_target(oof,'centered'),sample_weight=weights)
    return model

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    return np.clip(base+strength*detector.predict(frame[columns(kind)]),1e-6,1-1e-6)
