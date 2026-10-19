"""Research-only lagged daily execution strength, independently built from raw input."""
import numpy as np
import pandas as pd

COLUMNS=['execution_imbalance1','execution_imbalance5','execution_imbalance20','execution_change5']

def source_quality(raw,official):
    g=raw.copy();g['observed_date']=pd.to_datetime(g.dt,format='%Y%m%d')
    if g.observed_date.duplicated().any():raise ValueError('Duplicate source observation')
    for c in ['cntr_str','cur_prc']:
        g[c]=pd.to_numeric(g[c].astype(str).str.replace(',',''),errors='coerce')
    g['cur_prc']=g.cur_prc.abs()
    g=g.merge(official[['date','close']].rename(columns={'date':'observed_date','close':'official_close'}),on='observed_date',how='left',validate='one_to_one')
    # Missing/zero strength semantics unverified: quarantine rather than infer sell-only.
    g['valid']=g.cntr_str.gt(0)&np.isfinite(g.cntr_str)&g.cur_prc.eq(g.official_close)&g.official_close.gt(0)
    g['missing_official']=g.official_close.isna()
    g['price_mismatch']=g.official_close.notna()&~g.cur_prc.eq(g.official_close)
    g['invalid_strength']=~g.cntr_str.gt(0)|~np.isfinite(g.cntr_str)
    return g

def build(raw,official,calendar):
    dates=pd.DatetimeIndex(calendar).sort_values().unique()
    g=source_quality(raw,official)
    audit=dict(source_rows=len(g),nonpositive_or_missing_strength=int((~g.cntr_str.gt(0)|~np.isfinite(g.cntr_str)).sum()),price_mismatch_or_unverified=int((~g.cur_prc.eq(g.official_close)).sum()),valid_source_rows=int(g.valid.sum()))
    g=g.set_index('observed_date').reindex(dates)
    # Strength is a percentage ratio. This bounded transform avoids log explosions.
    imbalance=((g.cntr_str-100)/(g.cntr_str+100)).where(g.valid.eq(True))
    out=pd.DataFrame({'observed_date':dates,'execution_imbalance1':imbalance.to_numpy()})
    out['execution_imbalance5']=imbalance.rolling(5,min_periods=5).mean().to_numpy()
    out['execution_imbalance20']=imbalance.rolling(20,min_periods=20).mean().to_numpy()
    out['execution_change5']=imbalance.diff(5).to_numpy()
    mapping=dict(zip(dates[:-1],dates[1:]));out['date']=out.observed_date.map(mapping)
    out['usable']=np.isfinite(out[COLUMNS]).all(axis=1)
    out=out.dropna(subset=['date']).reset_index(drop=True)
    assert out.observed_date.lt(out.date).all()
    return out,audit
