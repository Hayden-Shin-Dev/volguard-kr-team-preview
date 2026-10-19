"""Date-common error removal for training; never centers on future test labels."""
import numpy as np
from lightgbm import LGBMRegressor
from .oof_signal import assert_oof
from .direction import COLS

COLUMNS=[c for c in COLS if c!='market_range']+['relative20','relative_rank','momentum_reversal','gap_body_disagreement','volume_x_reversal']

def residual_target(oof,kind):
    err=oof.label-oof.base_oof
    if kind=='centered':err=err-err.groupby(oof.date).transform('mean')
    return err.to_numpy()

def fit_detector(oof,kind):
    assert_oof(oof)
    weights=1/oof.date.map(oof.date.value_counts()).to_numpy();weights/=weights.mean()
    model=LGBMRegressor(objective='regression',n_estimators=120,learning_rate=.025,num_leaves=7,min_child_samples=1000,reg_lambda=20,n_jobs=2,random_state=1721,verbosity=-1)
    model.fit(oof[COLUMNS],residual_target(oof,kind),sample_weight=weights)
    return model

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    # No labels, no label-filtered cross-sectional normalization at inference.
    return np.clip(base+strength*detector.predict(frame[COLUMNS]),1e-6,1-1e-6)
