"""Disjoint rank feature family attribution with nested OOF residual targets."""
import numpy as np
from lightgbm import LGBMRegressor
from .oof_orthogonal import fit_detector as fit_absolute,COLUMNS
from .oof_relative import RANKS
from .oof_signal import assert_oof
GROUPS={
 'trend':['return_1','return_5','return_10','return_20','ma_deviation','relative20','relative_rank'],
 'candle':['gap','body_pct','close_location'],
 'activity':['vol_22','vol_change','range_1','range_5','range_22','volume_ratio','turnover_ratio'],
 'interaction':['momentum_reversal','gap_body_disagreement','volume_x_reversal']}
assert set(sum(GROUPS.values(),[]))==set(COLUMNS)
assert len(sum(GROUPS.values(),[]))==len(COLUMNS)

def fit_detector(oof,kind):
 assert_oof(oof);absolute=fit_absolute(oof,'centered')
 if kind=='absolute':return absolute,None,[]
 cols=RANKS if kind=='all' else ['rank_'+c for c in GROUPS[kind]]
 g=oof.dropna(subset=['absolute_oof']);assert g.absolute_fit_last_target.lt(g.date).all()
 y=g.label-g.base_oof-g.absolute_oof;y-=y.groupby(g.date).transform('mean');w=1/g.date.map(g.date.value_counts()).to_numpy();w/=w.mean()
 m=LGBMRegressor(objective='regression',n_estimators=80,learning_rate=.02,num_leaves=3,min_child_samples=2000,reg_lambda=40,n_jobs=2,random_state=1721,verbosity=-1)
 m.fit(g[cols],y,sample_weight=w);return absolute,m,cols

def combined_probability(base,detector,frame,kind,strength):
 if strength==0:return np.asarray(base)
 absolute,extra,cols=detector;value=base+absolute.predict(frame[COLUMNS])
 if extra is not None:value+=extra.predict(frame[cols])
 return np.clip(value,1e-6,1-1e-6)
