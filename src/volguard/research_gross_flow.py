"""Research-only gross participation from observed buy amounts and observed net amounts.

Sell magnitude is algebraically derived (buy minus net), not a directly fetched
sell series. The bounded three-way probe verifies the broker's signed-sell
convention, with at most one reported unit of rounding discrepancy in its sample.
Historical revision vintages remain unverified.
"""
import numpy as np
import pandas as pd
from .free_information_study import number

FIELDS={'foreign':'frgnr_invsr','institution':'orgn'}
COLUMNS=[f'{side}_{feature}{n}' for side in FIELDS for feature in ['participation','imbalance'] for n in [1,5,20]]

def features(buy,net,calendar):
    for frame in [buy,net]:
        if frame.dt.duplicated().any():raise ValueError('Duplicate source dates')
    joined=buy.merge(net,on='dt',suffixes=('_buy','_net'),validate='one_to_one')
    if joined.empty:return pd.DataFrame(columns=['date','observed_date']+COLUMNS)
    joined['observed_date']=pd.to_datetime(joined.dt,format='%Y%m%d');joined=joined.set_index('observed_date').sort_index()
    dates=pd.DatetimeIndex(calendar);joined=joined.reindex(dates[(dates>=joined.index.min())&(dates<=joined.index.max())])
    tb=number(joined.acc_trde_prica_buy);tn=number(joined.acc_trde_prica_net)
    # Never combine mismatched daily turnover or substitute missing values.
    turnover=tn.where(tn.gt(0)&tb.eq(tn));out=pd.DataFrame(index=joined.index)
    for side,field in FIELDS.items():
        b=number(joined[field+'_buy']);n=number(joined[field+'_net']);sell=b-n
        valid=turnover.notna()&b.ge(0)&sell.ge(0)&b.le(turnover+2)&sell.le(turnover+2)
        gross=(b+sell).where(valid);net_amount=n.where(valid);scale=turnover.where(valid)
        for window in [1,5,20]:
            total=gross.rolling(window).sum()
            out[f'{side}_participation{window}']=total/(2*scale.rolling(window).sum())
            out[f'{side}_imbalance{window}']=net_amount.rolling(window).sum()/total.where(total.gt(0))
    out['observed_date']=out.index;out['date']=out.index.map(dict(zip(dates[:-1],dates[1:])))
    return out.reset_index(drop=True).dropna(subset=['date'])[['date','observed_date']+COLUMNS]
