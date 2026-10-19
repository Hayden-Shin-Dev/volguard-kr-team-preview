"""Observed option activity ratios; neither order direction nor an implied probability."""
import re
import numpy as np
import pandas as pd

NAME = re.compile(r'^코스피200 ([CP]) (\d{6}) ([\d,.]+) \(정규\)$')
COLUMNS = ['log_put_call_volume', 'log_put_call_turnover', 'log_put_call_oi']


def summarize(rows):
    parsed=[]
    for r in rows:
        if r.get('PROD_NM') != '코스피200 옵션':
            continue
        match=NAME.fullmatch(r.get('ISU_NM',''))
        if not match:
            if '(야간)' in r.get('ISU_NM',''):
                continue
            raise ValueError('Unverified option name/session format')
        side, month, strike=match.groups()
        if r.get('RGHT_TP_NM') != ('CALL' if side=='C' else 'PUT'):
            raise ValueError('Option right mismatch')
        parsed.append(dict(date=r['BAS_DD'],symbol=r['ISU_CD'],side=side,month=month,strike=float(strike.replace(',','')),
            volume=r['ACC_TRDVOL'],turnover=r['ACC_TRDVAL'],oi=r['ACC_OPNINT_QTY']))
    if not parsed:
        return None
    f=pd.DataFrame(parsed)
    if f.date.nunique()!=1 or f.symbol.duplicated().any() or f.duplicated(['side','month','strike']).any():
        raise ValueError('Mixed dates or duplicate contracts')
    for c in ['volume','turnover','oi']:
        f[c]=pd.to_numeric(f[c].astype(str).str.replace(',',''),errors='coerce')
    if not np.isfinite(f[['volume','turnover','oi']]).all().all() or (f[['volume','turnover','oi']]<0).any().any():
        raise ValueError('Invalid activity fields')
    # Same-date most-traded expiry, chosen without future volume or returns.
    totals=f.groupby('month').volume.sum().sort_index()
    month=totals.idxmax()
    g=f[f.month==month]
    # Shared strikes prevent different listed strike sets determining the ratio.
    paired=g.groupby('strike').side.nunique()
    g=g[g.strike.isin(paired[paired==2].index)]
    if g.empty:
        return None
    a=g.groupby('side')[['volume','turnover','oi']].sum()
    if set(a.index)!={'C','P'} or (a<=0).any().any():
        return None
    result=dict(date=pd.Timestamp(f.date.iloc[0]),expiry_month=month,paired_strikes=int(g.strike.nunique()),
                call_volume=float(a.loc['C','volume']),put_volume=float(a.loc['P','volume']))
    for col,key in zip(COLUMNS,['volume','turnover','oi']):
        result[col]=float(np.log(a.loc['P',key]/a.loc['C',key]))
    return result


def lag_context(summary, calendar):
    if summary.date.duplicated().any():
        raise ValueError('Duplicate summary dates')
    f=summary.set_index('date').reindex(pd.DatetimeIndex(calendar).sort_values().unique())
    f['source_date']=f.index
    f=f.shift(1)
    f['usable']=np.isfinite(f[COLUMNS]).all(axis=1)
    f.index.name='date'
    return f.reset_index()
