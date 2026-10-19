"""Frozen-rule monthly walk-forward calibration research; never changes releases."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .direction import COLS
from .direction_scope_control import paired, strong
from .event_direction import freeze
from .platform_study import local_settings
from .probability_quality import evaluate, band
from .direction_historical_reliability import draws
from .util import atomic_json, digest, utcnow

RUN='adaptive-calibration-20261017'
METHODS=['fixed_existing','fixed_date_balanced','rolling_6m','rolling_12m','rolling_24m','weighted_12m','regime_12m','hierarchical_12m']


def logits(p):
    return logit(np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)).reshape(-1,1)


def eligible(history, cutoff, months=None):
    """Only confirmed outcomes; source timestamps remain conservative, not backdated."""
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None: raise ValueError('Timezone-aware cutoff required')
    available=pd.to_datetime(history.target_available_at,utc=True)
    mask=(available<=cutoff)&(history.issue_at<cutoff)
    if months is not None:
        mask &= history.issue_at >= cutoff-pd.DateOffset(months=months)
    return history.loc[mask].copy()


def weights(frame,cutoff=None,half_life_days=None):
    w=1/frame.target_date.map(frame.target_date.value_counts()).to_numpy(float)
    if half_life_days is not None:
        age=(pd.Timestamp(cutoff)-pd.to_datetime(frame.target_available_at,utc=True)).dt.total_seconds().to_numpy()/86400
        w*=np.exp2(-age/half_life_days)
    return w/w.sum()*len(frame)


def fit(frame, cutoff=None, half_life_days=None, date_balanced=True):
    if frame.target_date.nunique()<60 or len(frame)<1000 or frame.label.nunique()<2:
        return None
    w=weights(frame,cutoff,half_life_days) if date_balanced else np.ones(len(frame))
    model=LogisticRegression(C=1000,max_iter=500,random_state=1721)
    model.fit(logits(frame.raw_p),frame.label,sample_weight=w)
    if model.n_iter_.max()>=500: raise ValueError('Calibration failed to converge')
    return model


def predict(model,p):
    return np.clip(model.predict_proba(logits(p))[:,1],1e-6,1-1e-6)


def shrink_offset(frame, global_probability, prior_days=60):
    """Group intercept shrunk to pooled mapping; no confidence target or threshold."""
    n=frame.target_date.nunique()
    if n<60 or len(frame)<1000: return 0.0
    w=weights(frame); w=w/w.sum()
    z=logit(np.clip(global_probability,1e-6,1-1e-6))
    y=frame.label.to_numpy(float)
    penalty=prior_days/n
    delta=0.0
    for _ in range(50):
        p=expit(z+delta)
        step=(float(w@(p-y))+penalty*delta)/(float(w@(p*(1-p)))+penalty)
        delta-=step
        if abs(step)<1e-9:break
    return float(delta)


def prepare(raw, trained_before):
    f=raw.copy()
    f['issue_at']=(f.date+pd.Timedelta(hours=15,minutes=30)).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    f['target_available_at']=pd.to_datetime(f.target_available_at,utc=True)
    return f[f.issue_at>=pd.Timestamp(trained_before,tz='Asia/Seoul')].sort_values(['date','symbol']).reset_index(drop=True)


def walk(raw, target, fixed, fixed_balanced, threshold, group):
    """Refit on first forecast of each month; all methods share issue-date rows."""
    output=[]; audit=[]; snapshots={}
    for month,dayframe in target.groupby(target.date.dt.to_period('M'),sort=True):
        cutoff=dayframe.issue_at.min()
        h12=eligible(raw,cutoff,12)
        models={}; fallback={}
        for months in [6,12,24]:
            history=eligible(raw,cutoff,months)
            model=fit(history)
            key=f'rolling_{months}m'
            models[key]=model if model is not None else fixed_balanced
            fallback[key]=model is None
            audit.append(dict(month=str(month),method=key,cutoff=str(cutoff),rows=len(history),dates=int(history.target_date.nunique()),max_outcome_available_at=str(history.target_available_at.max()),first_issue=str(history.issue_at.min()),fallback=model is None))
        weighted=fit(h12,cutoff,183)
        models['weighted_12m']=weighted if weighted is not None else fixed_balanced
        fallback['weighted_12m']=weighted is None
        pooled=models['rolling_12m']
        regimes={}; offsets={}
        for high in [False,True]:
            h=h12[h12.market_range.ge(threshold).eq(high)]
            specific=fit(h)
            n=h.target_date.nunique()
            # Partial pooling: 60 pseudo-days of support for global calibration.
            regimes[high]=dict(model=specific,blend=float(n/(n+60)) if specific is not None else 0.,rows=len(h),dates=n)
        for member in [False,True]:
            h=h12[h12.symbol.isin(group).eq(member)]
            offsets[member]=shrink_offset(h,predict(pooled,h.raw_p)) if len(h) else 0.
        g=dayframe[['symbol','date','target_date','target_available_at','issue_at','label','raw_p','market_range']].copy()
        g['fixed_existing']=predict(fixed,g.raw_p)
        g['fixed_date_balanced']=predict(fixed_balanced,g.raw_p)
        for method,model in models.items():g[method]=predict(model,g.raw_p)
        pooled_prob=g.rolling_12m.to_numpy()
        regime_prob=pooled_prob.copy()
        for high,spec in regimes.items():
            mask=g.market_range.ge(threshold).eq(high).to_numpy()
            if spec['model'] is not None and mask.any():
                regime_prob[mask]=(1-spec['blend'])*pooled_prob[mask]+spec['blend']*predict(spec['model'],g.loc[mask,'raw_p'])
        g['regime_12m']=regime_prob
        delta=np.where(g.symbol.isin(group),offsets[True],offsets[False])
        g['hierarchical_12m']=expit(logit(np.clip(pooled_prob,1e-6,1-1e-6))+delta)
        output.append(g)
        snapshots[str(month)]=dict(cutoff=str(cutoff),models=models,regimes=regimes,group_offsets=offsets,fallback=fallback,group=group)
        audit.append(dict(month=str(month),method='weighted_regime_hierarchy',cutoff=str(cutoff),rows=len(h12),dates=int(h12.target_date.nunique()),max_outcome_available_at=str(h12.target_available_at.max()),weighted_fallback=weighted is None,regime_support={str(k):dict(rows=v['rows'],dates=int(v['dates']),blend=v['blend']) for k,v in regimes.items()},offsets={str(k):v for k,v in offsets.items()}))
        print('WALK',str(month),len(g),flush=True)
    return pd.concat(output,ignore_index=True),audit,snapshots


def run():
    root=local_settings().data_dir;out=root/'runs'/RUN;out.mkdir(exist_ok=True)
    feature_path=root/'runs/direction-calibration-study-20261017/features.parquet'
    original_path=root/'runs/direction-calibration-study-20261017/initial_refit__sigmoid.joblib'
    protected={str(p):digest(p.read_bytes()) for folder in ['config','direction-forward','forward'] for p in (root/folder).rglob('*') if p.is_file()}
    for p in root.rglob('*.joblib'):
        if out not in p.parents:protected[str(p)]=digest(p.read_bytes())
    protocol=dict(version=RUN,methods=METHODS,base_architecture='Existing initial COLS + imputer/scaler/logistic C=.1; no new features',tracks={'pre2023':'research-only base fitted through 2017, OOS calibration stream 2018+, fixed baseline 2019, evaluate issue dates 2020-2022 after 24-month warmup','diagnostic2024':'unchanged saved base fitted through 2021, OOS stream 2022+, saved fixed 2022 calibrator, 2024 diagnostics only after 24-month warmup'},selection='None. No year used to choose candidate, window, threshold, cadence, or gate. 2023/2024 never used for retrospective tuning. Earlier matured 2023/2024 outcomes enter ONLY the predeclared causal online update in diagnostic replay.',cadence='first forecast date of month at 15:30 KST; keep mapping until next month',windows='6/12/24 calendar months based on original forecast issue time; minimum 60 outcome dates and 1000 rows',weighted='12 month window, 183 calendar-day half-life; date-balanced weights',regime='12m, high/low market range relative to training-only median per date; local sigmoid shrunk to pooled with 60 prior days',hierarchy='12m pooled sigmoid + penalized cohort intercept, penalty 60/group dates; cohort top 30 training-period turnover incl three core; no sector or per-stock claims',fixed_controls='existing row-weighted fixed sigmoid plus fixed date-balanced sigmoid to isolate weighting',availability='Keep source target_available_at unchanged; <= first monthly cutoff required. Full 24m raw OOS history before comparison; no in-sample raw predictions admitted',probabilities='No quantile stretching or quota for high confidence. Only numerical clipping 1e-6',independent=False,production_change=False,source_hash=digest(Path(__file__).read_bytes()),features_hash=digest(feature_path.read_bytes()),base_hash=digest(original_path.read_bytes()),protected=protected)
    freeze(out/'protocol.json',protocol)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    f=pd.read_parquet(feature_path)
    bars=pd.read_parquet(root/'silver/bars.parquet',columns=['symbol','date','turnover'])
    reports={}
    with threadpool_limits(limits=3):
        for track,end,seedyear,evalyears in [('pre2023',2017,2019,[2020,2021,2022]),('diagnostic2024',2021,2022,[2024])]:
            dest=out/track;dest.mkdir(exist_ok=True)
            train=f[f.target_date.dt.year.le(end)]
            if track=='diagnostic2024':
                obj=joblib.load(original_path);base=obj['model'];fixed=obj['calibrator']
            else:
                base=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(C=.1,max_iter=700,random_state=17))
                base.fit(train[COLS],train.label)
                if base[-1].n_iter_.max()>=700:raise ValueError('Base failed to converge')
                joblib.dump(base,dest/'research-base.joblib')
                fixed=None
            raw=prepare(f,pd.Timestamp(year=end+1,month=1,day=1))
            raw=raw[raw.target_date.dt.year.le(max(evalyears))].copy()
            raw['raw_p']=base.predict_proba(raw[COLS])[:,1]
            start=raw.issue_at.min()+pd.DateOffset(years=2)
            target=raw[raw.target_date.dt.year.isin(evalyears)&raw.issue_at.ge(start)].copy()
            seed=eligible(raw,target.issue_at.min())
            seed=seed[seed.target_date.dt.year.eq(seedyear)]
            if fixed is None:fixed=fit(seed,date_balanced=False)
            fixed_balanced=fit(seed)
            if fixed is None or fixed_balanced is None:raise ValueError('Insufficient fixed seed')
            threshold=float(train.groupby('date').market_range.first().median())
            rank=bars[bars.date.dt.year.eq(end)].groupby('symbol').turnover.mean().sort_values(ascending=False)
            group=list(dict.fromkeys(['005930','000660','005380']+[s for s in rank.index if s in set(train.symbol)]))[:30]
            raw.to_parquet(dest/'raw-stream.parquet',index=False)
            predictions,audit,snapshots=walk(raw,target,fixed,fixed_balanced,threshold,group)
            predictions.to_parquet(dest/'predictions.parquet',index=False)
            joblib.dump(dict(fixed=fixed,fixed_date_balanced=fixed_balanced,monthly=snapshots),dest/'calibrators.joblib')
            atomic_json(dest/'availability-audit.json',audit)
            prior=float(train.label.mean())
            sections={}
            for period,g in [('all',predictions)]+[(str(y),predictions[predictions.target_date.dt.year.eq(y)]) for y in evalyears]:
                if len(g)==0:continue
                metrics={};contrasts={}
                dates=sorted(g.target_date.unique());indices=draws(dates)
                for method in METHODS:
                    p=g[method].to_numpy()
                    metrics[method]=dict(**evaluate(g,p,prior),strong=strong(g,p),bands60to70={side:band(g,p,side,.6,.7,indices,dates) for side in ['up','down']})
                    if method!='fixed_existing':
                        contrasts[method]=dict(vs_existing=paired(g,[(-1,g.fixed_existing.to_numpy()),(1,p)]),vs_fixed_date_balanced=paired(g,[(-1,g.fixed_date_balanced.to_numpy()),(1,p)]))
                sections[period]=dict(rows=len(g),dates=len(dates),results=metrics,contrasts=contrasts)
            reports[track]=dict(training_rows=len(train),training_symbols=int(train.symbol.nunique()),seed_rows=len(seed),market_threshold=threshold,cohort=group,first_issue=str(target.issue_at.min()),last_issue=str(target.issue_at.max()),monthly_updates=len(snapshots),periods=sections,availability_pass=all(pd.Timestamp(a['max_outcome_available_at'])<=pd.Timestamp(a['cutoff']) for a in audit))
            atomic_json(dest/'report.json',reports[track])
            print('TRACK_DONE',track,flush=True)
    unchanged=all(Path(p).exists() and digest(Path(p).read_bytes())==h for p,h in protected.items())
    result=dict(at=utcnow(),protocol={k:v for k,v in protocol.items() if k!='protected'},tracks=reports,production_unchanged=unchanged,independent=False,selected=None,artifact_dir=str(out),limitations=['Earlier periods also researched historically; no new independent test claimed','2024 only causal replay diagnostics; no candidate selection','Date-balanced block intervals do not remove repeated-study/multiple-comparison bias','Current survivor universe and conservative retrospective availability timestamps','Monthly updates may lag intra-month regime changes; cadence was not tuned','Hierarchy is pooled market/cohort, not per-stock or sector calibration'])
    atomic_json(out/'report.json',result);atomic_json(root/'reports/adaptive-calibration.json',result)
    print('DONE protected_unchanged',unchanged,flush=True)


if __name__=='__main__':run()
