"""Institution subtypes and full-market contrasts. Frozen-development research only."""
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from .platform_study import local_settings
from .flow_ablation import flow_features,DYNAMICS
from .free_information_study import number
from .direction import COLS
from .state_weighted_study import STATE
from .long_history_probability import estimator,positive_calibration,apply_cal
from .intraday_probability_study import report_metrics
from .direction_scope_control import paired
from .probability_quality import band
from .direction_historical_reliability import draws
from .util import atomic_json,digest

RUN='institution-components-20261017'
TYPES={'financial':'fnnc_invt','pension':'penfnd_etc','trust':'invtrt','insurance':'insrnc','private':'samo_fund','bank':'bank','total':'orgn'}
EDGES=[(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,.8),(.8,.9),(.9,1.)]
BASE=COLS+STATE

def columns(name):return [x.replace('foreign_',name+'_') for x in DYNAMICS('foreign')]

def summary(frame,p):
    y=frame.label.to_numpy();p=np.asarray(p);up=p>.5;down=p<.5;chosen=up|down
    hit=np.where(up,y,1-y)
    daily=pd.DataFrame({'date':frame.target_date.to_numpy(),'hit':hit,'keep':chosen}).query('keep').groupby('date').hit.mean()
    return dict(overall=float(hit[chosen].mean()) if chosen.any() else None,date_equal_accuracy=float(daily.mean()),up_accuracy=float(y[up].mean()) if up.any() else None,down_accuracy=float((1-y[down]).mean()) if down.any() else None,up_n=int(up.sum()),down_n=int(down.sum()),ties=int((~chosen).sum()),strong60=float((np.maximum(p,1-p)>=.6).mean()),strong70=float((np.maximum(p,1-p)>=.7).mean()))

def market_sample(root,out):
    calendar=sorted(pd.read_parquet(root/'runs/long-history-foundation-20261017/normalized-bars.parquet',columns=['date']).date.unique())
    parts=[];quality=[]
    for path in (root/'runs/flow-ablation-20261017').glob('flow-*.parquet'):
        raw=pd.read_parquet(path);joined=None
        for name,field in TYPES.items():
            modified=raw.copy();modified['frgnr_invsr']=raw[field]
            # All leading subtype zeros are unknown availability, not fabricated neutral data.
            active=number(raw[field]).ne(0).cummax();modified=modified.loc[active]
            ff=flow_features(modified,calendar)
            if ff.empty:continue
            keep=ff[['date']+DYNAMICS('foreign')].rename(columns=dict(zip(DYNAMICS('foreign'),columns(name))))
            joined=keep if joined is None else joined.merge(keep,on='date',validate='one_to_one')
        if joined is None:continue
        joined['symbol']=path.stem.split('-')[1];parts.append(joined)
        quality.append(dict(symbol=joined.symbol.iloc[0],rows=len(joined),complete=int(joined.dropna().shape[0])))
    allcols=sum([columns(n) for n in TYPES],[])
    flow=pd.concat(parts).dropna(subset=allcols)
    market=flow.groupby('date')[allcols].mean();market['reporting']=flow.groupby('date').size();market=market[market.reporting.ge(20)].reset_index()
    market.to_parquet(out/'sample-market-features.parquet',index=False);atomic_json(out/'data-quality.json',dict(stocks=quality,days=len(market),min_reporting=20,scope='fixed 60 survivor stocks; equal mean of subtype net/turnover ratios; not official full market'))
    return market,calendar

def official_features(out,calendar):
    raw=pd.read_parquet(out/'full-kospi-flow.parquet');g=raw.copy();g['date']=pd.to_datetime(g.date);g=g.sort_values('date').set_index('date')
    days=pd.DatetimeIndex(calendar);g=g.reindex(days[(days>=g.index.min())&(days<=g.index.max())]);f=pd.DataFrame(index=g.index)
    # Rolling normalization uses past 60 observations, not future standardization.
    for name,field in [('official_total','orgn_netprps'),('official_securities','sc_netprps'),('official_insurance','insrnc_netprps'),('official_trust','invtrt_netprps'),('official_fund','endw_netprps')]:
        v=number(g[field]);scale=v.abs().rolling(60,min_periods=60).mean().shift(1).replace(0,np.nan)
        f[name+'_scaled1']=v/scale;f[name+'_scaled5']=v.rolling(5).mean()/scale
        f[name+'_persistence']=np.sign(v).rolling(20).mean();f[name+'_acceleration']=v.diff()/scale
    mapping=dict(zip(days[:-1],days[1:]));f['date']=f.index.map(mapping)
    return f.reset_index(drop=True).dropna()

def fit_panel(f,groups,out,panel,condition=False):
    train=f[f.target_date.lt('2019-01-01')];tune=f[f.target_date.between('2019-01-01','2020-12-31')];cal=f[f.target_date.between('2021-01-01','2022-12-31')];review=f[f.target_date.between('2023-01-01','2024-12-31')].reset_index(drop=True)
    prior=float(train.label.mean());models={};scores={}
    thresholds=dict(flow=float(train.total_intensity1.abs().quantile(.9)),low=float(train.market_trend.quantile(.1)),high=float(train.market_trend.quantile(.9)))
    def event(g):return g.total_intensity1.abs().ge(thresholds['flow'])&(g.market_trend.le(thresholds['low'])|g.market_trend.ge(thresholds['high']))
    def predict(bundle,g):
        m,cols,conditional=bundle
        if not conditional:return m.predict_proba(g[cols])[:,1]
        q=predict(models['price'],g);mask=event(g)
        if mask.any():q[mask]=m.predict_proba(g.loc[mask,cols])[:,1]
        return q
    with threadpool_limits(limits=2):
        for name,extra in groups.items():
            cols=BASE+extra;m=estimator('lgb7');m.fit(train[cols],train.label);assert list(m.classes_)==[0,1]
            models[name]=(m,cols,False)
        event_train=train[event(train)]
        if condition and len(event_train)>=5000 and event_train.target_date.nunique()>=100:
            cols=BASE+groups['all_subtypes'];m=estimator('lgb7');m.fit(event_train[cols],event_train.label);models['conditional_shock']=(m,cols,True)
        for name,b in models.items():
            q=predict(b,tune);y=tune.label.to_numpy();d=pd.DataFrame(dict(date=tune.target_date,b=(q-y)**2,ll=-(y*np.log(np.clip(q,1e-9,1))+(1-y)*np.log(np.clip(1-q,1e-9,1))))).groupby('date').mean();scores[name]=[float(d.b.mean()),float(d.ll.mean())]
        chosen=min((x for x in models if x!='price'),key=scores.get);atomic_json(out/(panel+'-selection.json'),dict(selected=chosen,scores=scores,event_training_rows=len(event_train),event_training_dates=int(event_train.target_date.nunique()),thresholds=thresholds))
        pred=review[['symbol','date','target_date','label','market_trend']].copy();pred['shock_event']=event(review);results={}
        for name,b in models.items():
            raw=predict(b,review);z=positive_calibration(predict(b,cal),cal.label);p=apply_cal(z,raw);pred[name]=p;pred[name+'__raw']=raw
            results[name]=dict(raw=report_metrics(review,raw,prior),calibrated=report_metrics(review,p,prior),summary=summary(review,p),slope=float(np.exp(z[0])))
            joblib.dump(dict(model=b[0],columns=b[1],conditional=b[2],fallback=models['price'][0],thresholds=thresholds,calibration=z,research_only=True),out/(panel+'-'+name+'.joblib'))
            print('RESULT',panel,name,results[name]['calibrated']['date_balanced']['brier'],flush=True)
    pred.to_parquet(out/(panel+'-predictions.parquet'),index=False)
    dates=sorted(review.target_date.unique());ix=draws(dates)
    bands={n:{v:[band(review,pred[n+suffix].to_numpy(),side,lo,hi,ix,dates) for side in ['up','down'] for lo,hi in EDGES] for v,suffix in [('raw','__raw'),('calibrated','')]} for n in results}
    years={str(y):{n:dict(summary=summary(review[review.target_date.dt.year.eq(y)],pred.loc[review.target_date.dt.year.eq(y),n]),metrics=report_metrics(review[review.target_date.dt.year.eq(y)].reset_index(drop=True),pred.loc[review.target_date.dt.year.eq(y),n].to_numpy(),prior)) for n in results} for y in [2023,2024]}
    old=pd.read_parquet(out.parent/'long-history-foundation-20261017/predictions.parquet')[['symbol','target_date','label','all','frozen_initial']]
    shared=pred.merge(old,on=['symbol','target_date','label'],validate='one_to_one')
    reference={n:{k:paired(shared,[(-1,shared[k].to_numpy()),(1,shared[n].to_numpy())]) for k in ['all','frozen_initial']} for n in results}
    r=dict(selected=chosen,results=results,bands=bands,years=years,reference_contrasts=reference,contrasts={n:paired(review,[(-1,pred.price.to_numpy()),(1,pred[n].to_numpy())]) for n in results if n!='price'},splits={n:dict(rows=len(g),symbols=int(g.symbol.nunique()),dates=int(g.target_date.nunique()),first=str(g.target_date.min()),last=str(g.target_date.max())) for n,g in [('train',train),('tune',tune),('calibration',cal),('review',review)]},independent=False)
    atomic_json(out/(panel+'-report.json'),r);return r

def run():
    root=local_settings().data_dir;out=root/'runs'/RUN;out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Completed experiment preserved')
    protected={str(p):digest(p.read_bytes()) for p in root.rglob('*.joblib') if out not in p.parents}
    atomic_json(out/'protocol.json',dict(train='available through2018',selection='2019-2020 only; raw Brier then LL',calibration='2021-2022 same positive sigmoid',review='2023-2024 reused retrospective; not independent',groups=list(TYPES),cutoff='strict previous session; historical vintage not verified',event='training-only 90th percentile absolute institution flow and 10/90th market trend; min5000 rows and100 dates; pooled fallback',bands=EDGES,production_change=False))
    sample,calendar=market_sample(root,out)
    base=pd.read_parquet(root/'runs/long-history-foundation-20261017/features.parquet').merge(pd.read_parquet(root/'runs/state-weighted-direction-20261017/states.parquet'),on=['symbol','date'],validate='one_to_one')
    f=base.merge(sample,on='date',validate='many_to_one').dropna(subset=BASE).reset_index(drop=True)
    groups={'price':[],**{n:columns(n) for n in TYPES},'all_subtypes':sum([columns(n) for n in TYPES if n!='total'],[])}
    sample_report=fit_panel(f,groups,out,'sample',condition=True)
    official=official_features(out,calendar);matched=f.merge(official,on='date',validate='many_to_one').reset_index(drop=True)
    groups2={'price':[],'sample_total':columns('total'),'official_total':[c for c in official if c.startswith('official_total')],'official_subtypes':[c for c in official if c.startswith('official_') and not c.startswith('official_total')]}
    full_report=fit_panel(matched,groups2,out,'official',condition=False)
    atomic_json(out/'report.json',dict(sample=sample_report,official=full_report,protected_models=len(protected),unchanged=all(digest(Path(p).read_bytes())==h for p,h in protected.items()),independent=False))
    (out/'source.py').write_bytes(Path(__file__).read_bytes());print('DONE',flush=True)

if __name__=='__main__':run()
