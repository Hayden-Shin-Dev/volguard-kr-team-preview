"""Research-only time-ordered probability residuals. No production state access."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .direction import COLS

MARKET=['market_trend','market_vol','breadth','breadth5','market_range']
EXTRA=['relative20','relative_rank','momentum_reversal','gap_body_disagreement','relative_x_breadth','volume_x_reversal']

def enrich(frame):
    f=frame.copy()
    f['relative_rank']=f.groupby('date').relative20.rank(pct=True)-.5
    f['momentum_reversal']=f.return_1*f.return_20
    f['gap_body_disagreement']=f.gap*f.body_pct
    f['relative_x_breadth']=f.relative20*(f.breadth-.5)
    f['volume_x_reversal']=f.volume_ratio*f.return_1
    return f

def foundation():
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=17))

def up_probability(model,frame,columns):
    return model.predict_proba(frame[columns])[:,list(model.classes_).index(1)]

def assert_oof(frame):
    if frame.duplicated(['symbol','date']).any():raise ValueError('Duplicate OOF row')
    if not frame.fit_last_target.lt(frame.date).all():raise ValueError('OOF learned labels at/after prediction date')
    if not frame.target_date.gt(frame.date).all():raise ValueError('Invalid next-session label')
    if not frame.base_oof.between(0,1).all():raise ValueError('Invalid probability')

def detector_columns(kind):
    return list(dict.fromkeys(['base_oof']+MARKET+(COLS+EXTRA if kind=='stock' else [])))

def fit_detector(oof,kind):
    assert_oof(oof)
    cols=detector_columns(kind)
    weights=1/oof.date.map(oof.date.value_counts()).to_numpy();weights/=weights.mean()
    # Squared residual loss optimizes the Brier correction. Leaf support and L2
    # are fixed before development evaluation; no confidence-targeted weights.
    m=LGBMRegressor(objective='regression',n_estimators=120,learning_rate=.025,num_leaves=7,
                    min_child_samples=1000,reg_lambda=20,n_jobs=2,random_state=1721,verbosity=-1)
    m.fit(oof[cols],oof.label.to_numpy()-oof.base_oof.to_numpy(),sample_weight=weights)
    return m

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    x=frame.copy();x['base_oof']=base
    # Projection enforces valid probability bounds; it does not set a minimum
    # confidence. The learned signed residual treats both directions equally.
    return np.clip(base+strength*detector.predict(x[detector_columns(kind)]),1e-6,1-1e-6)
