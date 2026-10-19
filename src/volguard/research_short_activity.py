"""Lagged short-sale *transaction* activity; not outstanding short positions.

Research-only: a one-session lag does not establish historical publication or
revision timestamps. No API prices are used as labels or price features.
"""
import numpy as np
import pandas as pd
from volguard.free_information_study import number

COLUMNS=['short_share1','short_share5','short_share20','short_share_change5']

def features(raw,calendar):
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(calendar).unique()))
    f=raw.copy()
    f['observed_date']=pd.to_datetime(f.dt,format='%Y%m%d',errors='raise')
    if f.observed_date.duplicated().any():raise ValueError('Duplicate short activity dates')
    f=f.set_index('observed_date').reindex(sessions)
    for c in ['trde_qty','shrts_qty','trde_wght']:
        f[c]=number(f[c])
    calculated=100*f.shrts_qty/f.trde_qty
    valid=np.isfinite(f[['trde_qty','shrts_qty','trde_wght']]).all(axis=1)
    valid &= f.trde_qty.gt(0)&f.shrts_qty.ge(0)&f.shrts_qty.le(f.trde_qty)
    valid &= f.trde_wght.between(0,100)&(calculated-f.trde_wght).abs().le(.02)
    qty=f.shrts_qty.where(valid);total=f.trde_qty.where(valid)
    z=pd.DataFrame(index=sessions)
    z['short_share1']=calculated.where(valid)
    for n in [5,20]:
        z['short_share'+str(n)]=100*qty.rolling(n,min_periods=n).sum()/total.rolling(n,min_periods=n).sum()
    # Missing sessions stay missing, including no published transaction rows.
    z['short_share_change5']=(z.short_share1-z.short_share1.shift(5)).where(valid.rolling(6).sum().eq(6))
    z['observed_date']=sessions
    z['source_valid']=valid
    z=z.shift(1)
    z['date']=sessions
    z['usable']=np.isfinite(z[COLUMNS]).all(axis=1)
    return z.reset_index(drop=True)
