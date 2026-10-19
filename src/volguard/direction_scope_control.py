"""Controlled training/calibration scope factorial; research only, no releases."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .direction import COLS
from .direction_calibration_study import fit_calibration, calibrated
from .direction_historical_reliability import draws
from .event_direction import freeze
from .free_information_study import FLOW, DART
from .platform_study import local_settings
from .probability_quality import evaluate, losses
from .util import atomic_json, digest, utcnow

RUN = 'direction-scope-control-20261017'


def split_frames(full, small):
    """Same features/labels and dates; scope alone changes cross-sectional rows."""
    if full.duplicated(['symbol','target_date']).any() or small.duplicated(['symbol','target_date']).any():
        raise ValueError('Duplicate samples')
    check = small.merge(full[['symbol','target_date','label'] + COLS], on=['symbol','target_date'], how='left', validate='one_to_one', suffixes=('', '_full'))
    if not check.label.eq(check.label_full).all():
        raise ValueError('Target identity mismatch')
    if not np.allclose(check[COLS], check[[c+'_full' for c in COLS]], equal_nan=True):
        raise ValueError('Base features differ')
    masks = dict(train=lambda f: f.target_date.lt('2022-01-01'), calibration=lambda f: f.target_date.dt.year.eq(2022), review2023=lambda f: f.target_date.dt.year.eq(2023), review2024=lambda f: f.target_date.dt.year.eq(2024))
    scopes={}
    for name,mask in masks.items():
        s=small.loc[mask(small)].sort_values(['target_date','symbol']).reset_index(drop=True)
        a=full.loc[mask(full) & full.target_date.isin(s.target_date)].sort_values(['target_date','symbol']).reset_index(drop=True)
        if len(s)==0 or len(a)==0:
            raise ValueError('Empty split')
        scopes[name]={'full':a,'thirty':s}
    if scopes['train']['full'].target_date.max() >= scopes['calibration']['full'].target_date.min():
        raise ValueError('Training/calibration overlap')
    return scopes


def paired(frame, terms):
    """Linear contrast of losses; date blocks preserve same-day dependence."""
    dates=sorted(frame.target_date.unique())
    delta=np.zeros((len(frame),2))
    for coefficient,p in terms:
        delta += coefficient*np.column_stack(losses(frame.label.to_numpy(), p))
    daily=pd.DataFrame(delta,columns=['brier','log_loss']).assign(date=frame.target_date.to_numpy()).groupby('date').mean().reindex(dates)
    boot=daily.to_numpy()[draws(dates)].mean(axis=1)
    return {c:dict(delta=float(daily[c].mean()),ci95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,c in enumerate(daily.columns)}


def strong(frame,p):
    result={}
    for side,q,y in [('up',p,frame.label.to_numpy()),('down',1-p,1-frame.label.to_numpy())]:
        for name,mask in [('60plus',q>=.6),('60to70',(q>=.6)&(q<.7))]:
            f=frame.loc[mask]
            result[side+'_'+name]=dict(n=int(mask.sum()),dates=int(f.target_date.nunique()),coverage=float(mask.mean()),predicted=float(q[mask].mean()) if mask.any() else None,actual=float(y[mask].mean()) if mask.any() else None)
    return result


def run():
    root=local_settings().data_dir
    out=root/'runs'/RUN
    out.mkdir(exist_ok=True)
    source_full=root/'runs/direction-calibration-study-20261017/features.parquet'
    source_small=root/'runs/free-information-probability-20261017/features.parquet'
    full,small=pd.read_parquet(source_full),pd.read_parquet(source_small)
    scopes=split_frames(full,small)
    prior=float(scopes['train']['full'].label.mean())
    protected={str(p):digest(p.read_bytes()) for folder in ['config','forward','direction-forward'] for p in (root/folder).rglob('*') if p.is_file()}
    for p in root.rglob('*.joblib'):
        if out not in p.parents: protected[str(p)]=digest(p.read_bytes())
    protocol=dict(version=RUN,training='through 2021',calibration='2022 only',reviews=['2023 reused','2024 reused'],selection='None; fixed controlled contrasts, no evaluation-based candidate selection',estimator='imputer, standard scaler, logistic C=.1 max_iter=700 random_state=17; identical row weighting',calibrators=['sigmoid','beta'],factorial='base features: full/thirty training x full/thirty calibration',information='thirty training/thirty calibration only; same rows across base,flow,dart,combined; raw output also compared',date_control='Within each partition all scopes share exact target dates; smaller frame matches full base features and labels',inference_target='next regular close vs current close; conditional on non-flat',independent=False,source_hashes={str(source_full):digest(source_full.read_bytes()),str(source_small):digest(source_small.read_bytes())},code_hash=digest(Path(__file__).read_bytes()),prior_up=prior,protected=protected)
    freeze(out/'protocol.json',protocol)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    split_audit={k:{scope:dict(n=len(f),symbols=f.symbol.nunique(),dates=f.target_date.nunique(),first=str(f.target_date.min()),last=str(f.target_date.max()),up_rate=float(f.label.mean())) for scope,f in v.items()} for k,v in scopes.items()}
    atomic_json(out/'splits.json',split_audit)
    raw_models={}
    with threadpool_limits(limits=3):
        for name,scope,cols in [('full_base','full',COLS),('thirty_base','thirty',COLS),('thirty_flow','thirty',COLS+FLOW),('thirty_dart','thirty',COLS+DART),('thirty_combined','thirty',COLS+FLOW+DART)]:
            t=scopes['train'][scope]
            model=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=.1,max_iter=700,random_state=17))
            model.fit(t[cols],t.label)
            if model[-1].n_iter_.max()>=700: raise ValueError('Nonconverged fit')
            raw_models[name]=(model,cols)
            joblib.dump(dict(model=model,columns=cols,training_scope=scope),out/(name+'.joblib'))
            print('FIT',name,len(t),flush=True)
        recipes={}
        for name,(model,cols) in raw_models.items():
            recipes[name+'__raw']=(model,cols,None,'raw')
            for scope in (['full','thirty'] if name.endswith('_base') else ['thirty']):
                cal=scopes['calibration'][scope]
                raw=model.predict_proba(cal[cols])[:,1]
                for method in ['sigmoid','beta']:
                    c=fit_calibration(cal,raw,method)
                    key=f'{name}__cal_{scope}__{method}'
                    recipes[key]=(model,cols,c,method)
                    joblib.dump(dict(calibrator=c,calibration_scope=scope,method=method,parent_model=name),out/(key+'.joblib'))
    reports={}
    for year in [2023,2024]:
        f=scopes[f'review{year}']['thirty']
        preds=f[['symbol','date','target_date','label']].copy()
        results={}
        with threadpool_limits(limits=3):
            for name,(model,cols,c,method) in recipes.items():
                p=calibrated(c,f,model.predict_proba(f[cols])[:,1],method)
                preds[name]=p
                results[name]=dict(**evaluate(f,p,prior),strong=strong(f,p))
                print('EVAL',year,name,results[name]['date_balanced']['brier'],flush=True)
        contrasts={}
        def diff(label,a,b):
            contrasts[label]=dict(reference=a,candidate=b,**paired(f,[(-1,preds[a].to_numpy()),(1,preds[b].to_numpy())]))
        for method in ['sigmoid','beta']:
            for cal in ['full','thirty']:
                diff(f'training_effect__cal_{cal}__{method}',f'full_base__cal_{cal}__{method}',f'thirty_base__cal_{cal}__{method}')
            for train in ['full','thirty']:
                diff(f'calibration_scope_effect__train_{train}__{method}',f'{train}_base__cal_full__{method}',f'{train}_base__cal_thirty__{method}')
            terms=[(1,preds[f'thirty_base__cal_thirty__{method}'].to_numpy()),(-1,preds[f'full_base__cal_thirty__{method}'].to_numpy()),(-1,preds[f'thirty_base__cal_full__{method}'].to_numpy()),(1,preds[f'full_base__cal_full__{method}'].to_numpy())]
            contrasts[f'interaction__{method}']=paired(f,terms)
        for method in ['raw','sigmoid','beta']:
            suffix='__raw' if method=='raw' else f'__cal_thirty__{method}'
            for group in ['flow','dart','combined']:
                diff(f'information_effect__{group}__{method}','thirty_base'+suffix,'thirty_'+group+suffix)
        for train in ['full','thirty']:
            for cal in ['full','thirty']:
                diff(f'calibration_method__{train}__{cal}',f'{train}_base__cal_{cal}__sigmoid',f'{train}_base__cal_{cal}__beta')
        preds.to_parquet(out/f'predictions-{year}.parquet',index=False)
        reports[str(year)]=dict(results=results,contrasts=contrasts)
    unchanged=all(Path(p).exists() and digest(Path(p).read_bytes())==h for p,h in protected.items())
    report=dict(at=utcnow(),protocol={k:v for k,v in protocol.items() if k!='protected'},splits=split_audit,years=reports,production_unchanged=unchanged,independent=False,auto_promotion=False,limitations=['New information available for 30 names only; no full-universe information factorial', 'Current survivor universe; flow original vintages not certified', 'Retrospective diagnostic; correlated multiple comparisons, not promotion evidence','Scope includes sample count and composition; those two are not isolated'],artifact_dir=str(out))
    atomic_json(out/'report.json',report)
    atomic_json(root/'reports/direction-scope-control.json',report)
    print('DONE protected_unchanged',unchanged,flush=True)


if __name__=='__main__': run()
