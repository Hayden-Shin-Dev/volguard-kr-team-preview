"""Frozen state-conditioned learning research. Never writes production artifacts."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from .platform_study import local_settings
from .direction import COLS
from .long_history_probability import estimator, positive_calibration, apply_cal
from .free_information_study import number
from .intraday_probability_study import report_metrics
from .direction_scope_control import paired
from .probability_quality import band
from .direction_historical_reliability import draws
from .util import atomic_json, digest

STATE=['market_trend','market_vol','breadth','breadth5','relative20']
RUN='state-weighted-direction-20261017'

def state_features(raw):
    """State membership never depends on a next-day label being available."""
    parts=[]
    days=pd.DatetimeIndex(sorted(raw.date.unique()))
    for sym,g in raw.groupby('symbol'):
        g=g.sort_values('date').set_index('date').reindex(days)
        ret=g.close.pct_change(fill_method=None)
        ret=ret.where(g.valid.fillna(False)&g.valid.shift().fillna(False)&ret.abs().le(.4))
        parts.append(pd.DataFrame({'date':days,'symbol':sym,'ret':ret.to_numpy(),'r20':np.expm1(np.log1p(ret).rolling(20).sum()).to_numpy()}))
    x=pd.concat(parts,ignore_index=True)
    market=x.groupby('date').agg(mret=('ret','mean'),breadth=('ret',lambda s:s.dropna().gt(0).mean()),count=('ret','count'))
    market['market_trend']=np.expm1(np.log1p(market.mret).rolling(20).sum())
    market['market_vol']=market.mret.rolling(20).std()
    market['breadth5']=market.breadth.rolling(5).mean()
    x=x.merge(market[STATE[:-1]],left_on='date',right_index=True,validate='many_to_one')
    x['relative20']=x.r20-x.market_trend
    return x[['symbol','date']+STATE]

def flow_states(root, dates):
    pieces=[]
    for path in (root/'runs/free-information-probability-20261017').glob('flow-*.parquet'):
        g=pd.read_parquet(path).sort_values('dt')
        scale=number(g.acc_trde_prica).where(lambda x:x>0)
        active=pd.concat([number(g[c]) for c in ['frgnr_invsr','orgn','ind_invsr']],axis=1).abs().sum(axis=1).gt(0).cummax()
        z=pd.DataFrame({'observed':pd.to_datetime(g.dt),'foreign':number(g.frgnr_invsr)/scale,'institution':number(g.orgn)/scale})
        pieces.append(z[active])
    flow=pd.concat(pieces).groupby('observed').agg(foreign=('foreign','mean'),institution=('institution','mean'),coverage=('foreign','count')).sort_index()
    flow=flow[flow.coverage.ge(20)].reset_index()
    for c in ['foreign','institution']:flow[c]=flow[c].rolling(5).mean()
    result=pd.merge_asof(pd.DataFrame({'date':sorted(dates)}),flow,left_on='date',right_on='observed',direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta(days=7))
    assert (result.observed.dropna()<result.loc[result.observed.notna(),'date']).all()
    return result

def fit_panel(f,cols,out,panel):
    train=f[f.target_date.lt('2019-01-01')].copy()
    cal=f[f.target_date.between('2021-01-01','2022-12-31')].copy()
    review=f[f.target_date.between('2023-01-01','2024-12-31')].copy()
    assert train.target_date.max()<cal.date.min() and cal.target_date.max()<review.target_date.min()
    features=COLS+cols
    base=estimator('lgb7');base.fit(train[COLS],train.label)
    full=estimator('lgb7');full.fit(train[features],train.label)
    # Fit state scaler and centers only on historical training data, balanced by date.
    w=1/train.target_date.map(train.target_date.value_counts()).to_numpy()
    scaler=StandardScaler().fit(train[cols],sample_weight=w)
    z=scaler.transform(train[cols]); km=KMeans(n_clusters=4,n_init=10,random_state=17).fit(z,sample_weight=w)
    experts=[];weighted=[];audit=[]
    for k in range(4):
        mask=km.labels_==k;g=train[mask]
        d=np.mean((z-km.cluster_centers_[k])**2,axis=1)
        sw=np.exp(-d/2);sw=np.maximum(sw,.02)
        # ESS counts dates, not correlated stocks, for fallback decisions.
        dw=pd.DataFrame({'date':train.target_date,'w':sw}).groupby('date').w.mean()
        ess=float(dw.sum()**2/(dw@dw))
        expert=estimator('lgb7') if len(g)>=5000 and g.target_date.nunique()>=250 and g.label.nunique()==2 else None
        if expert is not None:expert.fit(g[features],g.label)
        wm=estimator('lgb7') if ess>=250 else None
        if wm is not None:wm.fit(train[features],train.label,**{wm.steps[-1][0]+'__sample_weight':sw})
        experts.append(expert);weighted.append(wm)
        audit.append(dict(cluster=k,rows=len(g),dates=int(g.target_date.nunique()),weight_effective_dates=ess,expert_fallback=expert is None,weighted_fallback=wm is None,center=dict(zip(cols,scaler.inverse_transform(km.cluster_centers_[[k]])[0].tolist()))))
        print('FIT',panel,k,len(g),flush=True)
    # Neighbor search on historical daily market state plus relative-return terciles.
    rel_edges=np.quantile(train.relative20,[1/3,2/3])
    groups=np.searchsorted(rel_edges,train.relative20)
    tmp=train[cols+['date','target_date','label']].copy();tmp['relative_group']=groups
    grouped=tmp.groupby(['date','relative_group']).agg({**{c:'mean' for c in cols},'label':'mean','target_date':'first'}).reset_index()
    nn=NearestNeighbors(n_neighbors=min(1500,len(grouped))).fit(scaler.transform(grouped[cols]))
    def infer(q):
        # Frozen estimators; state routing computed separately at every historical close.
        a=scaler.transform(q[cols]);distance=((a[:,None,:]-km.cluster_centers_[None,:,:])**2).mean(axis=2)
        route=np.argmin(distance,axis=1);mix=np.exp(-(distance-distance.min(axis=1,keepdims=True))/2);mix/=mix.sum(axis=1,keepdims=True)
        p0=base.predict_proba(q[COLS])[:,1];pf=full.predict_proba(q[features])[:,1]
        pe=pf.copy();pw=np.zeros(len(q))
        for k in range(4):
            mask=route==k
            if experts[k] is not None and mask.any():pe[mask]=experts[k].predict_proba(q.loc[mask,features])[:,1]
            pw+=mix[:,k]*(weighted[k].predict_proba(q[features])[:,1] if weighted[k] is not None else pf)
        # Compute once per date x relative tercile, then map to all matching stocks.
        query=q[cols+['date']].copy();query['relative_group']=np.searchsorted(rel_edges,q.relative20)
        query=query.groupby(['date','relative_group'])[cols].mean().reset_index()
        dist,ids=nn.kneighbors(scaler.transform(query[cols]));probs=[];neff=[]
        for dd,ii in zip(dist,ids):
            hist=grouped.iloc[ii];keep=~hist.date.duplicated();ii=ii[keep][:250];dd=dd[keep][:250]
            ww=np.exp(-dd**2/(2*max(float(np.median(dd)),.1)**2));eff=float(ww.sum()**2/(ww@ww));neff.append(eff)
            probs.append(float(ww@grouped.label.to_numpy()[ii]/ww.sum()) if eff>=100 else np.nan)
        query['near']=probs;query['near_ess']=neff
        joined=q[['date','relative20']].assign(relative_group=np.searchsorted(rel_edges,q.relative20)).merge(query[['date','relative_group','near','near_ess']],on=['date','relative_group'],how='left',validate='many_to_one')
        pn=joined.near.to_numpy();pn=np.where(np.isfinite(pn),pn,pf)
        return dict(base=p0,state_full=pf,similar_weighted=pw,experts=pe,neighbors=pn),route,joined.near_ess.to_numpy()
    cp,_,_=infer(cal);rp,route,ess=infer(review)
    result={};pred=review[['symbol','date','target_date','label']+cols].copy();pred['state']=route;pred['neighbor_ess']=ess
    prior=float(train.label.mean())
    for key in rp:
        cz=positive_calibration(cp[key],cal.label);pred[key+'__raw']=rp[key];pred[key]=apply_cal(cz,rp[key])
        result[key]=dict(raw=report_metrics(review,rp[key],prior),calibrated=report_metrics(review,pred[key].to_numpy(),prior),slope=float(np.exp(cz[0])))
    pred.to_parquet(out/(panel+'-predictions.parquet'),index=False)
    joblib.dump(dict(base=base,full=full,experts=experts,weighted=weighted,scaler=scaler,clusters=km,relative_edges=rel_edges,neighbor_history=grouped,columns=features),out/(panel+'-research.joblib'))
    contrasts={key:paired(review,[(-1,pred.base.to_numpy()),(1,pred[key].to_numpy())]) for key in rp if key!='base'}
    subgroup={}
    for k in range(4):
        mask=route==k;g=review[mask].reset_index(drop=True)
        subgroup[str(k)]={key:report_metrics(g,pred.loc[mask,key].to_numpy(),prior) for key in rp}
    years={str(y):{key:report_metrics(review[review.target_date.dt.year.eq(y)].reset_index(drop=True),pred.loc[review.target_date.dt.year.eq(y),key].to_numpy(),prior) for key in rp} for y in [2023,2024]}
    # Include all >=70% predictions, rather than omitting the 80-100% tail.
    bands={key:[band(review,pred[key].to_numpy(),side,lo,hi,draws(sorted(review.target_date.unique())),sorted(review.target_date.unique())) for side in ['up','down'] for lo,hi in [(.55,.6),(.6,.65),(.65,.7),(.7,1.)]] for key in rp}
    return dict(results=result,contrasts=contrasts,states=audit,subgroups=subgroup,years=years,bands=bands,splits={k:dict(n=len(g),dates=int(g.target_date.nunique()),first=str(g.target_date.min()),last=str(g.target_date.max())) for k,g in [('train',train),('cal',cal),('review',review)]})

def run():
    root=local_settings().data_dir;out=root/'runs'/RUN;out.mkdir(exist_ok=True)
    if (out/'report.json').exists():raise RuntimeError('Completed research preserved')
    protected={str(p):digest(p.read_bytes()) for p in root.rglob('*.joblib') if out not in p.parents}
    atomic_json(out/'protocol.json',dict(train='all available through 2018',calibration='2021-2022 fixed positive sigmoid, no method search',review='2023-2024 frozen-model sequential walk-forward with state-dependent routing at each close; reused retrospective, not independent',selection='none, all fixed contrasts; no review-driven model selection',state='current survivor universe equal-weight price proxies; not KOSPI200 official breadth',flow='30-stock sample mean prior-date flow; >=20 reporting stocks; historical revisions unverified; exploratory only',new='multi-dimensional similarity weights + mixture of weighted experts + unique-date nearest analogs',overlap='previous Markov volatility experts and technical event conditions already tested',neighbor_rule='250 unique historical dates, effective dates >=100 else pooled fallback; target labels always <=2018',experts='4 unsupervised training-only states; >=5000 rows and >=250 dates else pooled fallback',production_changed=False))
    f=pd.read_parquet(root/'runs/long-history-foundation-20261017/features.parquet')
    raw=pd.read_parquet(root/'runs/long-history-foundation-20261017/normalized-bars.parquet')
    state=state_features(raw);state.to_parquet(out/'states.parquet',index=False)
    f=f.merge(state,on=['symbol','date'],validate='one_to_one').dropna(subset=STATE).reset_index(drop=True)
    flow=flow_states(root,raw.date.unique());flow.to_parquet(out/'flow-state.parquet',index=False)
    flowf=f.merge(flow,on='date',validate='many_to_one').dropna(subset=['foreign','institution']).reset_index(drop=True)
    with threadpool_limits(limits=2):
        results={}
        for name,frame,cols in [('price_state',f,STATE),('flow_available_control',flowf,STATE),('flow_state',flowf,STATE+['foreign','institution'])]:
            print('PANEL',name,len(frame),flush=True)
            results[name]=fit_panel(frame,cols,out,name);atomic_json(out/(name+'-report.json'),results[name])
    a=pd.read_parquet(out/'flow_available_control-predictions.parquet');b=pd.read_parquet(out/'flow_state-predictions.parquet')
    pd.testing.assert_frame_equal(a[['symbol','target_date','label']],b[['symbol','target_date','label']])
    flowcontrast={k:paired(a,[(-1,a[k].to_numpy()),(1,b[k].to_numpy())]) for k in ['state_full','similar_weighted','experts','neighbors']}
    atomic_json(out/'report.json',dict(panels=results,flow_contrasts=flowcontrast,protected_models=len(protected),models_unchanged=all(digest(Path(p).read_bytes())==h for p,h in protected.items()),independent=False))
    print('DONE',out,flush=True)

if __name__=='__main__':run()
