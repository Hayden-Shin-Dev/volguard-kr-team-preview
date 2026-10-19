"""Conservative, calendar-aligned ownership features. Research only, one-session lag."""
import numpy as np
import pandas as pd
from volguard.free_information_study import number

COLUMNS=['ownership_pct','ownership_change5','ownership_change20','holding_change_fraction','ownership_pressure5']

def features(raw, calendar):
    sessions=pd.DatetimeIndex(sorted(pd.to_datetime(calendar).unique()))
    f=raw.copy()
    f['observed_date']=pd.to_datetime(f.dt,format='%Y%m%d')
    if f.observed_date.duplicated().any():raise ValueError('Duplicate ownership dates')
    f=f.set_index('observed_date').reindex(sessions)
    for c in ['poss_stkcnt','chg_qty','frgnr_limit','gain_pos_stkcnt','wght','limit_exh_rt']:
        f[c]=number(f[c])
    finite=np.isfinite(f[['poss_stkcnt','chg_qty','frgnr_limit','gain_pos_stkcnt','wght','limit_exh_rt']]).all(axis=1)
    valid=finite & f.poss_stkcnt.ge(0)&f.gain_pos_stkcnt.ge(0)&f.frgnr_limit.gt(0)&f.wght.between(0,100)
    valid &= (f.poss_stkcnt+f.gain_pos_stkcnt-f.frgnr_limit).abs().le(1)
    valid &= (100*f.poss_stkcnt/f.frgnr_limit-f.limit_exh_rt).abs().le(.02)
    # A gap cannot be interpreted as a one-session holding change.
    prior=f.poss_stkcnt.shift().notna() & f.frgnr_limit.shift().notna()
    residual=f.poss_stkcnt.diff()-f.chg_qty
    event=prior & (residual.abs().gt(1)|f.frgnr_limit.diff().ne(0))
    clean=valid & prior & ~event
    z=pd.DataFrame(index=sessions)
    z['ownership_pct']=f.wght.where(clean)
    z['holding_change_fraction']=(100*f.chg_qty/f.frgnr_limit).where(clean)
    for n in [5,20]:
        continuous=clean.rolling(n+1).sum().eq(n+1)
        z['ownership_change'+str(n)]=(f.wght-f.wght.shift(n)).where(continuous)
    z['ownership_pressure5']=z.holding_change_fraction.rolling(5,min_periods=5).sum()
    z['observed_date']=sessions
    z['source_valid']=valid
    z['quarantined_transition']=event
    # Fixed KRX session lag. It is not proof of original publication time.
    z=z.shift(1)
    z['date']=sessions
    z['usable']=np.isfinite(z[COLUMNS]).all(axis=1)
    return z.reset_index(drop=True)
