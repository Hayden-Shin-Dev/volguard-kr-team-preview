"""Research-only credit financing statistics, not same-day order flow.

Kiwoom's provision ratio is settlement-date based. A two-session lag is an
explicit conservative experiment convention, NOT verified publication history.
No balance differences, source prices, or cross-unit ratios are used.
"""
import numpy as np
import pandas as pd
from volguard.free_information_study import number

COLUMNS=['credit_balance_pct','credit_provision_pct','credit_imbalance1','credit_imbalance5']

def features(raw,calendar):
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(calendar).unique()))
    f=raw.copy();f['observed_date']=pd.to_datetime(f.dt,format='%Y%m%d',errors='raise')
    if f.observed_date.duplicated().any():raise ValueError('Duplicate credit dates')
    f=f.set_index('observed_date').reindex(sessions)
    fields=['new','rpya','pre','remn','shr_rt','remn_rt']
    for c in fields:f[c]=number(f[c])
    valid=np.isfinite(f[fields]).all(axis=1)
    valid &= f[['new','rpya','remn']].ge(0).all(axis=1)
    valid &= f.shr_rt.between(0,100)&f.remn_rt.between(0,100)
    valid &= (f['new']-f.rpya-f.pre).abs().le(1)
    net=(f['new']-f.rpya).where(valid);gross=(f['new']+f.rpya).where(valid)
    z=pd.DataFrame(index=sessions)
    z['credit_balance_pct']=f.remn_rt.where(valid)
    z['credit_provision_pct']=f.shr_rt.where(valid)
    z['credit_imbalance1']=net/gross.where(gross.gt(0))
    total=gross.rolling(5,min_periods=5).sum()
    z['credit_imbalance5']=net.rolling(5,min_periods=5).sum()/total.where(total.gt(0))
    z['observed_date']=sessions;z['source_valid']=valid
    z=z.shift(2);z['date']=sessions;z['usable']=np.isfinite(z[COLUMNS]).all(axis=1)
    return z.reset_index(drop=True)
