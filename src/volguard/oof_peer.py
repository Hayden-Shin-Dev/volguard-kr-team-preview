"""Frozen pre-2013 price co-movement network; not sector/causal attribution."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .oof_orthogonal import fit_detector as fit_absolute,COLUMNS
from .oof_signal import assert_oof
PEERS=['peer_return1','peer_return5','peer_relative20','peer_volume','peer_gap1','peer_gap5']

def build_network_features(allrows):
    early=allrows[allrows.date.lt('2013-01-01')].pivot(index='date',columns='symbol',values='return_1')
    corr=early.corr(min_periods=252);network={}
    for symbol in corr:
        candidates=corr[symbol].drop(index=symbol).dropna();candidates=candidates[candidates>0].sort_values(ascending=False)
        if len(candidates)>=5:network[symbol]=candidates.head(5).index.tolist()
    matrices={c:allrows.pivot(index='date',columns='symbol',values=c) for c in ['return_1','return_5','relative20','volume_ratio']};parts=[]
    for symbol,neighbors in network.items():
        own=allrows[allrows.symbol.eq(symbol)].set_index('date');g=pd.DataFrame(index=own.index)
        for source,dest in zip(matrices,['peer_return1','peer_return5','peer_relative20','peer_volume']):
            x=matrices[source].reindex(index=g.index,columns=neighbors)
            g[dest]=x.mean(axis=1).where(x.notna().sum(axis=1)>=4)
        g['peer_gap1']=g.peer_return1-own.return_1;g['peer_gap5']=g.peer_return5-own.return_5;g['symbol']=symbol;parts.append(g.reset_index())
    return pd.concat(parts,ignore_index=True),network

def fit_detector(oof,kind):
    assert_oof(oof);absolute=fit_absolute(oof,'centered')
    if kind=='absolute':return absolute,None
    g=oof.dropna(subset=['absolute_oof']+PEERS)
    assert g.absolute_fit_last_target.lt(g.date).all()
    y=g.label-g.base_oof-g.absolute_oof;y-=y.groupby(g.date).transform('mean');w=1/g.date.map(g.date.value_counts()).to_numpy();w/=w.mean()
    m=LGBMRegressor(objective='regression',n_estimators=80,learning_rate=.02,num_leaves=3,min_child_samples=2000,reg_lambda=40,n_jobs=2,random_state=1721,verbosity=-1)
    cols=PEERS if kind=='peer' else PEERS[:4]
    m.fit(g[cols],y,sample_weight=w);return absolute,(m,cols)

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    absolute,extra=detector;value=base+absolute.predict(frame[COLUMNS])
    if extra is not None:
        m,cols=extra;valid=frame[PEERS].notna().all(axis=1).to_numpy()
        # Missing peer history means NO extra correction, not invented zero features.
        if valid.any():value[valid]+=m.predict(frame.loc[valid,cols])
    return np.clip(value,1e-6,1-1e-6)
