"""Investor flow ablation: lagged inputs, fixed development selection, no release."""
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .platform_study import local_settings
from .free_information_study import number
from .long_history_probability import estimator,positive_calibration,apply_cal
from .direction import COLS
from .intraday_probability_study import report_metrics
from .direction_scope_control import paired
from .direction_historical_reliability import draws
from .probability_quality import band
from .util import atomic_json,digest

RUN='flow-ablation-20261017'
STRENGTH=lambda side:[side+'_intensity'+str(n) for n in [1,5,20]]
DYNAMICS=lambda side:STRENGTH(side)+[side+'_persistence5',side+'_persistence20',side+'_acceleration1',side+'_acceleration5']
GROUPS={'price':[], 'foreign_strength':STRENGTH('foreign'),'foreign_dynamics':DYNAMICS('foreign'),'institution_strength':STRENGTH('institution'),'institution_dynamics':DYNAMICS('institution'),'both_strength':STRENGTH('foreign')+STRENGTH('institution'),'both_dynamics':DYNAMICS('foreign')+DYNAMICS('institution'),'both_interactions':DYNAMICS('foreign')+DYNAMICS('institution')+['foreign_trend','institution_trend','flow_agreement']}
PRICE=COLS+['market_trend','relative20']

def flow_features(raw,calendar,lag=1):
    """Only previous completed sessions; missing sessions stay missing."""
    g=raw.copy();g['observed_date']=pd.to_datetime(g.dt);g=g.sort_values('observed_date').drop_duplicates('observed_date')
    active=pd.concat([number(g[c]) for c in ['frgnr_invsr','orgn','ind_invsr']],axis=1).abs().sum(axis=1).gt(0).cummax()
    g=g.loc[active].set_index('observed_date')
    if g.empty:return pd.DataFrame()
    dates=pd.DatetimeIndex(calendar);g=g.reindex(dates[(dates>=g.index.min())&(dates<=g.index.max())])
    scale=number(g.acc_trde_prica).where(lambda x:x>0)
    out=pd.DataFrame(index=g.index)
    for side,col in [('foreign','frgnr_invsr'),('institution','orgn')]:
        net=number(g[col]);intensity=(net/scale).where(lambda x:x.abs()<=1.01)
        net=net.where(intensity.notna())
        for n in [1,5,20]:out[side+'_intensity'+str(n)]=net.rolling(n).sum()/scale.where(net.notna()).rolling(n).sum()
        for n in [5,20]:out[side+'_persistence'+str(n)]=np.sign(intensity).rolling(n).mean()
        out[side+'_acceleration1']=intensity.diff()
        out[side+'_acceleration5']=out[side+'_intensity5']-out[side+'_intensity20']
    mapping=dict(zip(dates[:-lag],dates[lag:]))
    out['observed_date']=out.index;out['date']=out.index.map(mapping)
    return out.reset_index(drop=True).dropna(subset=['date'])

def build(root,out):
    base=pd.read_parquet(root/'runs/long-history-foundation-20261017/features.parquet')
    states=pd.read_parquet(root/'runs/state-weighted-direction-20261017/states.parquet')[['symbol','date','market_trend','relative20']]
    base=base.merge(states,on=['symbol','date'],validate='one_to_one')
    calendar=sorted(pd.read_parquet(root/'runs/long-history-foundation-20261017/normalized-bars.parquet',columns=['date']).date.unique())
    parts=[];audit=[]
    for path in sorted(out.glob('flow-*.parquet')):
        raw=pd.read_parquet(path);f=flow_features(raw,calendar);symbol=path.stem.split('-')[1]
        if f.empty:continue
        f['symbol']=symbol;parts.append(f)
        audit.append(dict(symbol=symbol,raw_rows=len(raw),usable_rows=int(f.dropna().shape[0]),first=str(f.dropna().date.min()),last=str(f.dropna().date.max())))
    flow=pd.concat(parts,ignore_index=True);f=base.merge(flow,on=['symbol','date'],validate='one_to_one')
    f['foreign_trend']=f.foreign_intensity5*f.market_trend;f['institution_trend']=f.institution_intensity5*f.market_trend
    f['flow_agreement']=f.foreign_intensity5*f.institution_intensity5
    f=f.replace([np.inf,-np.inf],np.nan).dropna(subset=PRICE+GROUPS['both_interactions']).sort_values(['target_date','symbol']).reset_index(drop=True)
    assert (f.observed_date<f.date).all()
    f.to_parquet(out/'features.parquet',index=False);atomic_json(out/'data-audit.json',dict(symbols=audit,total_rows=len(f),source_timing='prior-session only; current historical vintage, not certified PIT'))
    return f

def run():
    root=local_settings().data_dir;out=root/'runs'/RUN
    if (out/'report.json').exists():raise RuntimeError('Preserve completed study')
    protected={str(p):digest(p.read_bytes()) for p in root.rglob('*.joblib') if out not in p.parents}
    atomic_json(out/'protocol.json',dict(train='through 2018, all available flow observations',selection='2019-2020 raw date-balanced Brier, then LogLoss; groups fixed before review',calibration='2021-2022 fixed positive sigmoid; no calibration search',review='2023-2024 reused retrospective; parameters fixed, causal sequential predictions',price_baseline='full-universe all-history frozen model and matched-row price/context refit',cutoff='regular close; only previous session flow, no same-day finalized flows',timing_limitation='Historical receipt timestamps and revision history absent; exploratory, not verified point-in-time',groups=GROUPS,production_change=False))
    f=build(root,out);train=f[f.target_date.lt('2019-01-01')];tune=f[f.target_date.between('2019-01-01','2020-12-31')];cal=f[f.target_date.between('2021-01-01','2022-12-31')];review=f[f.target_date.between('2023-01-01','2024-12-31')].reset_index(drop=True)
    models={};scores={};prior=float(train.label.mean())
    with threadpool_limits(limits=2):
        for name,extra in GROUPS.items():
            cols=PRICE+extra;m=estimator('lgb7');m.fit(train[cols],train.label)
            q=m.predict_proba(tune[cols])[:,1];y=tune.label.to_numpy()
            d=pd.DataFrame(dict(date=tune.target_date,b=(q-y)**2,ll=-(y*np.log(np.clip(q,1e-9,1))+(1-y)*np.log(np.clip(1-q,1e-9,1))))).groupby('date').mean()
            scores[name]=[float(d.b.mean()),float(d.ll.mean())];models[name]=(m,cols)
            print('FIT',name,len(train),scores[name],flush=True)
        chosen=min((k for k in scores if k!='price'),key=scores.get)
        atomic_json(out/'selection.json',dict(chosen=chosen,scores=scores,selection_only='2019-2020',promoted=False))
        pred=review[['symbol','date','target_date','label','market_trend','relative20','observed_date']].copy();results={}
        frozen=joblib.load(root/'runs/long-history-foundation-20261017/all.joblib')
        p=apply_cal(frozen['calibration'],frozen['model'].predict_proba(review[frozen['columns']])[:,1]);pred['full_history']=p;results['full_history']=dict(calibrated=report_metrics(review,p,prior))
        for name,(m,cols) in models.items():
            cz=positive_calibration(m.predict_proba(cal[cols])[:,1],cal.label);raw=m.predict_proba(review[cols])[:,1];p=apply_cal(cz,raw)
            pred[name]=p;pred[name+'__raw']=raw
            joblib.dump(dict(model=m,columns=cols,calibration=cz,flow_lag=1,cutoff='previous session',train_end=str(train.target_date.max()),cal_end=str(cal.target_date.max()),research_only=True),out/(name+'.joblib'))
            results[name]=dict(raw=report_metrics(review,raw,prior),calibrated=report_metrics(review,p,prior),slope=float(np.exp(cz[0])))
    pred.to_parquet(out/'predictions.parquet',index=False)
    dates=sorted(review.target_date.unique());idx=draws(dates)
    bands={name:[band(review,pred[name].to_numpy(),side,lo,hi,idx,dates) for side in ['up','down'] for lo,hi in [(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,1.)]] for name in results}
    contrasts={name:paired(review,[(-1,pred.price.to_numpy()),(1,pred[name].to_numpy())]) for name in results if name!='price'}
    years={str(y):{name:report_metrics(review[review.target_date.dt.year.eq(y)].reset_index(drop=True),pred.loc[review.target_date.dt.year.eq(y),name].to_numpy(),prior) for name in results} for y in [2023,2024]}
    atomic_json(out/'report.json',dict(selected=chosen,selection=scores,results=results,bands=bands,contrasts_vs_matched_price=contrasts,years=years,splits={n:dict(rows=len(g),dates=int(g.target_date.nunique()),symbols=int(g.symbol.nunique()),first=str(g.target_date.min()),last=str(g.target_date.max())) for n,g in [('train',train),('tune',tune),('calibration',cal),('review',review)]},models_unchanged=all(digest(Path(p).read_bytes())==h for p,h in protected.items()),protected_models=len(protected),independent=False))
    (out/'source.py').write_bytes(Path(__file__).read_bytes());print('DONE',chosen,flush=True)

if __name__=='__main__':run()
