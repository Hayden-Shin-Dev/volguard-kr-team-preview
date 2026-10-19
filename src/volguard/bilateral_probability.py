"""Symmetric, frozen-prediction probability audit. Never selects a release."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .platform_study import local_settings
from .probability_quality import band, losses
from .direction_historical_reliability import draws
from .util import atomic_json, digest, utcnow

MODELS=['initial_refit__sigmoid','technical_recent__market_sigmoid']
BANDS=[(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,.8),(.6,.7)]

def audit(frame):
    dates=sorted(frame.target_date.unique()); idx=draws(dates)
    p=frame.p_up.to_numpy(); y=frame.label.to_numpy()
    result=[]
    for side in ['up','down']:
        q=p if side=='up' else 1-p
        actual=y if side=='up' else 1-y
        for lo,hi in BANDS:
            r=band(frame,p,side,lo,hi,idx,dates)
            mask=(q>=lo)&(q<hi)&(q>.5)
            # Exactly 50% is ambiguous, not two direction signals.
            if lo==.5 and np.any(q==.5):
                subset=frame[q!=.5]
                if len(subset):
                    r=band(subset,subset.p_up.to_numpy(),side,lo,hi,idx,dates)
                else:
                    # Empty signal set, retaining the calendar for bootstrap.
                    r=band(frame,np.full(len(frame),.49 if side=='up' else .51),side,lo,hi,idx,dates)
                r['coverage']=float(mask.mean())
            if mask.any():
                b,ll=losses(actual[mask],q[mask])
                r.update(brier=float(b.mean()),log_loss=float(ll.mean()),gap=float(actual[mask].mean()-q[mask].mean()))
            else:r.update(brier=None,log_loss=None,gap=None)
            r['mean_close']=bool(r['n'] and abs(r['gap'])<=.05)
            r['research_label']='회고 정합 근거' if r['status']=='compatible' else '평균 근접·검증 부족' if r['mean_close'] else '평균 불일치' if r['n'] else '신호 없음'
            result.append(r)
    return dict(rows=len(frame),dates=len(dates),ties=int(np.sum(p==.5)),bands=result)

def run():
    root=local_settings().data_dir
    source=root/'runs/direction-calibration-study-20261017'
    out=root/'runs/bilateral-probability-20261017';out.mkdir(exist_ok=True)
    protected={str(p):digest(p.read_bytes()) for folder in ['config','direction-forward','forward'] for p in (root/folder).rglob('*') if p.is_file()}
    for p in root.rglob('*.joblib'):protected[str(p)]=digest(p.read_bytes())
    models={}; hashes={}
    for model in MODELS:
        periods={}
        for year,prefix in [(2023,'development'),(2024,'final')]:
            path=source/f'{prefix}-{model}.parquet';hashes[str(path)]=digest(path.read_bytes())
            f=pd.read_parquet(path)
            periods[str(year)]=audit(f)
            for half in [1,2]:
                g=f[((f.target_date.dt.month-1)//6+1).eq(half)]
                periods[f'{year}-H{half}']=audit(g)
        summary=[]
        for side in ['up','down']:
            for lo,hi in BANDS:
                select=lambda period:next(b for b in periods[period]['bands'] if b['side']==side and b['low']==lo and b['high']==hi)
                years=[select(str(y)) for y in [2023,2024]]
                halves=[select(f'{y}-H{h}') for y in [2023,2024] for h in [1,2]]
                summary.append(dict(side=side,low=lo,high=hi,both_year_means_close=all(b['mean_close'] for b in years),all_half_means_close=all(b['mean_close'] for b in halves),all_half_supported=all(b['status']=='compatible' for b in halves),annual=years,halves=halves))
        models[model]=dict(periods=periods,stability=summary)
        print('AUDIT',model,flush=True)
    result=dict(at=utcnow(),models=models,independent=False,production_changed=False,selection=None,source_hashes=hashes,protected_unchanged=all(digest(Path(p).read_bytes())==h for p,h in protected.items()),policy='Both directions identical fixed bins; mean-close within 5pp is descriptive ONLY; original 500 rows/60 dates/concentration/95% gap CI gates unchanged; halves diagnostic; reused 2023/2024 never used to tune or promote; no new fitting; no conditional-range claim')
    atomic_json(out/'report.json',result);atomic_json(root/'reports/bilateral-probability.json',result)
    history=root/'reports/probability-quality-history.json'
    old=json.loads(history.read_text('utf8'));old['bilateral']=result;atomic_json(history,old)
    print('DONE unchanged',result['protected_unchanged'],flush=True)

if __name__=='__main__':run()
