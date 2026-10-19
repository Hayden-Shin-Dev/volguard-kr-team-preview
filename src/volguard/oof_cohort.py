"""Balance observable OOF confidence cohorts within dates, not outcome classes."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .oof_peer import PEERS,combined_probability
from .oof_orthogonal import fit_detector as fit_absolute
from .oof_signal import assert_oof

def cohort_weights(frame,smoothing):
    probability=np.clip(frame.base_oof.to_numpy()+frame.absolute_oof.to_numpy(),0,1)
    # Symmetric, fixed before evaluation. No minimum output confidence.
    cohort=np.searchsorted([.3,.4,.5,.6,.7],probability,side='right')
    g=pd.DataFrame({'date':frame.date.to_numpy(),'cohort':cohort})
    count=g.groupby(['date','cohort']).cohort.transform('size').to_numpy()
    w=1/(count+smoothing)
    totals=pd.Series(w).groupby(g.date).transform('sum').to_numpy()
    w/=totals;return w/w.mean()

def fit_detector(oof,kind):
    assert_oof(oof);absolute=fit_absolute(oof,'centered')
    if kind=='absolute':return absolute,None
    g=oof.dropna(subset=['absolute_oof']+PEERS)
    assert g.absolute_fit_last_target.lt(g.date).all()
    y=g.label-g.base_oof-g.absolute_oof;y-=y.groupby(g.date).transform('mean')
    if kind=='peer':w=1/g.date.map(g.date.value_counts()).to_numpy();w/=w.mean()
    else:w=cohort_weights(g,0 if kind=='cohort' else 20)
    m=LGBMRegressor(objective='regression',n_estimators=80,learning_rate=.02,num_leaves=3,min_child_samples=2000,reg_lambda=40,n_jobs=2,random_state=1721,verbosity=-1)
    m.fit(g[PEERS],y,sample_weight=w);return absolute,(m,PEERS)
