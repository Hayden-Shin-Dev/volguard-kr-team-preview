"""Long daily foundation and out-of-time minute residual research. No releases."""
import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from .platform_study import local_settings
from .direction import COLS
from .intraday_probability_study import report_metrics,MINUTE
from .direction_scope_control import paired
from .util import atomic_json,digest,utcnow

RUN='long-history-foundation-20261017'

def positive_calibration(p,y):
    """Strictly positive temperature preserves ordering; no direction flipping."""
    x=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y)
    def loss(z):
        score=np.exp(z[0])*x+z[1]
        return np.mean(np.logaddexp(0,score)-y*score)+.001*(z@z)
    r=minimize(loss,[0.,0.],method='L-BFGS-B',bounds=[(-4,4),(-5,5)])
    if not r.success:raise ValueError('Calibration failed')
    return r.x

def apply_cal(z,p):return expit(np.exp(z[0])*logit(np.clip(p,1e-6,1-1e-6))+z[1])

def features(raw,calendar=None):
    """Per-symbol ratios; all rolling windows are backward looking."""
    parts=[]
    if calendar is None:calendar=pd.DatetimeIndex(sorted(raw.date.unique()))
    for symbol,g in raw.groupby('symbol'):
        g=g.sort_values('date').copy()
        days=calendar[(calendar>=g.date.min())&(calendar<=g.date.max())]
        g=g.set_index('date').reindex(days).rename_axis('date').reset_index()
        g['symbol']=symbol;g['valid']=g.valid.fillna(False).astype(bool)
        c=g.close;ret=c.pct_change(fill_method=None)
        # Unverified corporate jumps are excluded, never interpolated.
        invalid=(~g.valid)|ret.abs().gt(.4)
        ret=ret.mask(invalid|invalid.shift(fill_value=False))
        x=g[['symbol','date']].copy();x['return_1']=ret
        for n in [5,10,20]:x[f'return_{n}']=np.expm1(np.log1p(ret).rolling(n).sum())
        x['vol_22']=ret.rolling(22).std();x['vol_change']=ret.rolling(5).std()/x.vol_22
        x['ma_deviation']=c/c.rolling(20).mean()-1
        span=100*(g.high/g.low-1)
        for n in [1,5,22]:x[f'range_{n}']=span.rolling(n).mean()
        x['gap']=g.open/c.shift()-1;x['body_pct']=100*(c/g.open-1)
        x['close_location']=(c-g.low)/(g.high-g.low).replace(0,np.nan)
        x.loc[g.high.eq(g.low),'close_location']=.5
        x['volume_ratio']=g.volume/g.volume.rolling(20).mean();x['turnover_ratio']=g.turnover/g.turnover.rolling(20).mean()
        x['target_date']=g.date.shift(-1);x['label']=ret.shift(-1).gt(0).astype(int);x['target_return']=ret.shift(-1)
        x['usable']=~invalid.rolling(22,min_periods=22).max().fillna(1).astype(bool)
        parts.append(x)
    f=pd.concat(parts).replace([np.inf,-np.inf],np.nan)
    market=f.groupby('date').range_1.mean();f['market_range']=f.date.map(market)
    return f

def dataset(root,out):
    parts=[];audit=[]
    for path in sorted((root/'runs/expanded-daily-20261017').glob('*.parquet')):
        g=pd.read_parquet(path);m={'dt':'date','cur_prc':'close','open_pric':'open','high_pric':'high','low_pric':'low','trde_qty':'volume','trde_prica':'turnover'};g=g.rename(columns=m)
        g['date']=pd.to_datetime(g.date,format='%Y%m%d');g['symbol']=path.stem
        for c in ['open','high','low','close','volume','turnover']:g[c]=pd.to_numeric(g[c],errors='coerce').abs()
        g=g.sort_values('date').drop_duplicates('date')
        g['valid']=g[['open','high','low','close']].gt(0).all(axis=1)&g.volume.gt(0)&g.high.ge(g[['open','close','low']].max(axis=1))&g.low.le(g[['open','close','high']].min(axis=1))
        # Provider adjustment-event markers are conservatively excluded with
        # the following feature warmup. No invented corporate-action factors.
        events=g.upd_stkpc_tp.fillna('').astype(str).str.strip().ne('')
        g['valid'] &= ~events
        audit.append(dict(symbol=path.stem,n=len(g),first=str(g.date.min().date()),last=str(g.date.max().date()),invalid=int((~g.valid).sum()),sha=digest(path.read_bytes())))
        parts.append(g[['symbol','date','open','high','low','close','volume','turnover','valid']])
    raw=pd.concat(parts)
    krx=pd.read_parquet(root/'silver/bars.parquet',columns=['date'])
    days=sorted(set(raw.date)|set(krx.date));f=features(raw,pd.DatetimeIndex(days))
    market=pd.read_parquet(root/'silver/market.parquet').drop_duplicates('date').set_index('date').range_pct
    f['market_range']=f.date.map(market).fillna(f.market_range)
    # Authoritative known dates in KRX archive; before that, union is only a proxy.
    krx=pd.read_parquet(root/'silver/bars.parquet',columns=['date']);days=sorted(set(raw.date)|set(krx.date));nxt=dict(zip(days[:-1],days[1:]))
    f['next_session_ok']=f.date.map(nxt).eq(f.target_date)
    valid=f.usable&f.next_session_ok&f.target_return.notna()&f.target_return.ne(0)&f[COLS].notna().all(axis=1)
    f=f[valid].sort_values(['target_date','symbol']).reset_index(drop=True)
    f.to_parquet(out/'features.parquet',index=False);raw.to_parquet(out/'normalized-bars.parquet',index=False);atomic_json(out/'data-audit.json',dict(symbols=audit,valid_feature_rows=len(f),raw_rows=len(raw),calendar_before2010='observed union, not certified exchange calendar',corporate_actions='provider adjustment flag requested; independent factor/vintage history unavailable; exclude invalid OHLC and >40% return transitions; residual risk remains',survivorship='current project universe only; delisted and historical membership incomplete'))
    return f

def estimator(kind):
    if kind=='logistic':e=LogisticRegression(C=.1,max_iter=800)
    else:e=LGBMClassifier(n_estimators=180,num_leaves=7 if kind=='lgb7' else 15,min_child_samples=150,learning_rate=.035,n_jobs=2,verbosity=-1,random_state=17)
    return make_pipeline(SimpleImputer(),StandardScaler(),e)

def run():
    root=local_settings().data_dir;out=root/'runs'/RUN;out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Completed study preserved')
    protected={str(p):digest(p.read_bytes()) for d in ['config','forward','direction-forward'] for p in (root/d).rglob('*') if p.is_file()}
    atomic_json(out/'protocol.json',dict(at=utcnow(),train='through 2018',tune='2019-2020',calibration='2021-2022',review='2023-2024 chronological fixed-parameter walk-forward, reused retrospective',recent5='2014-2018 target dates; warmup earlier',selection='Choose structure/window by tuning Brier then Log Loss ONLY, before calibration and review',calibration_method='positive slope logistic temperature only',production_change=False,policy='2025-2026 only auxiliary research; not independent; future forecasts not backfilled'))
    f=dataset(root,out);train=f[f.target_date.lt('2019-01-01')];tune=f[f.target_date.between('2019-01-01','2020-12-31')];cal=f[f.target_date.between('2021-01-01','2022-12-31')];test=f[f.target_date.between('2023-01-01','2024-12-31')]
    windows={'all':None,'since2000':'2000-01-01','since2010':'2010-01-01','recent5':'2014-01-01','all_recent_weight':None};selection={};models={};counts={}
    with threadpool_limits(limits=2):
        for window,start in windows.items():
            t=train if start is None else train[train.target_date.ge(start)]
            counts[window]=dict(rows=len(t),symbols=t.symbol.nunique(),dates=t.target_date.nunique(),first=str(t.target_date.min()),last=str(t.target_date.max()),warmup='22 actual observations; excluded missing/invalid windows')
            weights=np.exp(-np.log(2)*(pd.Timestamp('2018-12-31')-t.target_date).dt.days/1826.25) if window=='all_recent_weight' else np.ones(len(t))
            for kind in ['logistic','lgb7','lgb15']:
                key=window+'__'+kind;m=estimator(kind);m.fit(t[COLS],t.label,**{m.steps[-1][0]+'__sample_weight':weights})
                p=m.predict_proba(tune[COLS])[:,list(m.classes_).index(1)];daily=pd.DataFrame(dict(date=tune.target_date,b=(p-tune.label)**2,ll=-(tune.label*np.log(np.clip(p,1e-6,1))+(1-tune.label)*np.log(np.clip(1-p,1e-6,1))))).groupby('date').mean()
                selection[key]=[float(daily.b.mean()),float(daily.ll.mean())];models[key]=m;print('FIT',key,len(t),flush=True)
    chosen={w:min([k for k in selection if k.startswith(w+'__')],key=lambda k:selection[k]) for w in windows};global_choice=min(selection,key=lambda k:selection[k]);atomic_json(out/'selection.json',dict(by_window=chosen,overall=global_choice,scores=selection))
    results={};pred=test[['symbol','date','target_date','label']].copy();prior=float(train.label.mean())
    for w,key in chosen.items():
        m=models[key];z=positive_calibration(m.predict_proba(cal[COLS])[:,1],cal.label);p=apply_cal(z,m.predict_proba(test[COLS])[:,1]);pred[w]=p
        results[w]=dict(model=key,calibration_slope=float(np.exp(z[0])),overall=report_metrics(test,p,prior),years={str(y):report_metrics(test[test.target_date.dt.year.eq(y)],p[test.target_date.dt.year.eq(y)],prior) for y in [2023,2024]})
        joblib.dump(dict(model=m,calibration=z,columns=COLS),out/(w+'.joblib'));print('REVIEW',w,results[w]['overall']['date_balanced']['brier'],flush=True)
    # Frozen long-trained initial reference; identical labels and rows.
    from .direction_calibration_study import calibrated
    old=joblib.load(root/'runs/direction-calibration-study-20261017/initial_refit__sigmoid.joblib');raw=old['model'].predict_proba(test[old['cols']])[:,1];p=calibrated(old['calibrator'],test,raw,old['method']);pred['frozen_initial']=p;results['frozen_initial']=dict(overall=report_metrics(test,p,prior),feature_basis_note='Provider-adjusted OHLC reconstruction; market range is project-universe proxy; compare controlled refits for window effect')
    pred.to_parquet(out/'predictions.parquet',index=False)
    subgroup={}
    regimes={}
    regime_cut=float(train.market_range.median())
    for w in windows:
        values=[]
        for symbol,g in test.groupby('symbol'):
            ix=test.index.get_indexer(g.index);q=pred[w].to_numpy()[ix]
            values.append(dict(symbol=symbol,n=len(g),brier=float(np.mean((q-g.label)**2)),accuracy=float(np.mean((q>=.5)==g.label))))
        subgroup[w]=values
        regimes[w]={name:report_metrics(test[mask],pred[w].to_numpy()[mask.to_numpy()],prior) for name,mask in [('lower_range',test.market_range.le(regime_cut)),('higher_range',test.market_range.gt(regime_cut))]}
    atomic_json(out/'by-symbol.json',subgroup)
    atomic_json(out/'by-regime.json',dict(training_median_range=regime_cut,results=regimes))
    contrast={w:paired(test,[(-1,pred.since2010.to_numpy()),(1,pred[w].to_numpy())]) for w in windows}
    result=dict(selected_on_2019_2020=global_choice,counts=counts,results=results,contrasts_vs_since2010=contrast,production_unchanged=all(digest(Path(k).read_bytes())==v for k,v in protected.items()),splits={n:dict(rows=len(g),dates=g.target_date.nunique(),symbols=g.symbol.nunique()) for n,g in [('tune',tune),('cal',cal),('review',test)]},independent=False)
    atomic_json(out/'report.json',result);print('DONE',flush=True)

if __name__=='__main__':run()
