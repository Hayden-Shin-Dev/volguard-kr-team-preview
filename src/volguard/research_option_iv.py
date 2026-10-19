"""Traded regular-session IV shape, not a risk-neutral or physical direction probability."""
import numpy as np
import pandas as pd
from .research_options_context import NAME

COLUMNS=['log_atm_iv','put_wing_relative','call_wing_relative']

def summarize(options, futures):
    spot_rows=[r for r in futures if r.get('PROD_NM')=='코스피200 선물' and r.get('MKT_NM')=='정규']
    if not spot_rows:return None,'missing_spot'
    s=pd.DataFrame(spot_rows)
    for c in ['SPOT_PRC','ACC_TRDVOL']:
        s[c]=pd.to_numeric(s[c].astype(str).str.replace(',',''),errors='coerce')
    s=s[(s.SPOT_PRC>0)&(s.ACC_TRDVOL>0)]
    if s.empty:return None,'missing_spot'
    if s.BAS_DD.nunique()!=1:raise ValueError('Mixed futures dates')
    spot=float(s.loc[s.ACC_TRDVOL.idxmax(),'SPOT_PRC'])
    if float(s.SPOT_PRC.max()-s.SPOT_PRC.min())>.001*spot:return None,'inconsistent_spot'
    records=[]
    for r in options:
        if r.get('PROD_NM')!='코스피200 옵션':continue
        match=NAME.fullmatch(r.get('ISU_NM',''))
        if not match:
            if '(야간)' in r.get('ISU_NM',''):continue
            raise ValueError('Unverified session format')
        side,month,k=match.groups()
        if r['RGHT_TP_NM']!=('CALL' if side=='C' else 'PUT'):raise ValueError('Right mismatch')
        records.append(dict(date=r['BAS_DD'],side=side,month=month,strike=float(k.replace(',','')),
            volume=r['ACC_TRDVOL'],oi=r['ACC_OPNINT_QTY'],close=r['TDD_CLSPRC'],iv=r['IMP_VOLT']))
    f=pd.DataFrame(records)
    if f.empty:return None,'missing_options'
    if f.date.nunique()!=1 or f.date.iloc[0]!=s.BAS_DD.iloc[0]:raise ValueError('Source date mismatch')
    if f.duplicated(['side','month','strike']).any():raise ValueError('Duplicate contract')
    for c in ['volume','oi','close','iv']:
        f[c]=pd.to_numeric(f[c].astype(str).str.replace(',',''),errors='coerce')
    month=f.groupby('month').volume.sum().sort_index().idxmax()
    g=f[(f.month==month)&(f[['volume','oi','close','iv']]>0).all(axis=1)].copy()
    g=g[g.strike.between(spot*.9,spot*1.1)]
    paired=g.pivot(index='strike',columns='side',values='iv').reindex(columns=['C','P']).dropna()
    if len(paired)<3:return None,'insufficient_traded_pairs'
    atm_k=float((paired.index.to_series()/spot-1).abs().idxmin())
    if abs(atm_k/spot-1)>.025:return None,'atm_too_far'
    put=paired[paired.index<spot];call=paired[paired.index>spot]
    if put.empty or call.empty:return None,'missing_wing'
    put_k=float((put.index.to_series()/spot-.95).abs().idxmin())
    call_k=float((call.index.to_series()/spot-1.05).abs().idxmin())
    if abs(put_k/spot-.95)>.025 or abs(call_k/spot-1.05)>.025:return None,'wing_too_far'
    atm=float(paired.loc[atm_k].mean());put_iv=float(paired.loc[put_k,'P']);call_iv=float(paired.loc[call_k,'C'])
    return dict(date=pd.Timestamp(f.date.iloc[0]),spot=spot,expiry_month=month,paired_strikes=len(paired),
        atm_strike=atm_k,put_strike=put_k,call_strike=call_k,atm_iv=atm,put_iv=put_iv,call_iv=call_iv,
        atm_call_put_gap=float(abs(paired.loc[atm_k,'C']-paired.loc[atm_k,'P'])),
        log_atm_iv=float(np.log(atm)),put_wing_relative=put_iv/atm-1,call_wing_relative=call_iv/atm-1),'ok'

def lag_context(summary,calendar):
    if summary.date.duplicated().any():raise ValueError('Duplicate IV dates')
    f=summary.set_index('date').reindex(pd.DatetimeIndex(calendar).sort_values().unique())
    f['source_date']=f.index
    f=f.shift(1);f['usable']=np.isfinite(f[COLUMNS]).all(axis=1);f.index.name='date'
    return f.reset_index()
