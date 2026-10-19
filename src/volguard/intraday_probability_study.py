"""New close-to-next-close research fits; isolated from all production models."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits
from .platform_study import local_settings
from .data import Lake, features
from .direction import COLS, extend
from .direction_study import enhance
from .technical_features import build, GROUPS
from .probability_quality import evaluate, losses
from .direction_scope_control import paired
from .util import atomic_json, digest, utcnow

SYMBOLS=['005930','000660','005380']
RUN='intraday-probability-refit-20261017'
MINUTE=['rv','down_variation','path_efficiency','ret_autocorr','late_return','early_return','late_volume_share','vwap_gap','signed_volume_proxy','late_vs_early','rv_ratio']

def minute_features(raw):
    f=raw.copy();f['ts']=pd.to_datetime(f.cntr_tm,format='%Y%m%d%H%M%S')
    for c in ['cur_prc','open_pric','high_pric','low_pric','trde_qty']:
        f[c]=pd.to_numeric(f[c],errors='coerce').abs()
    f=f.sort_values('ts').drop_duplicates('ts');f['date']=f.ts.dt.normalize()
    clock=f.ts.dt.strftime('%H%M')
    f=f[clock.between('0900','1530') & f.cur_prc.gt(0) & f.trde_qty.ge(0)]
    rows=[]
    for date,g in f.groupby('date'):
        # Normal sessions only; delayed special sessions are excluded, not truncated.
        if g.ts.iloc[0].strftime('%H%M')!='0900' or g.ts.iloc[-1].strftime('%H%M')!='1530' or len(g)<300:continue
        p=g.cur_prc.to_numpy(); v=g.trde_qty.to_numpy();r=np.diff(np.log(p)); late=g.ts.dt.strftime('%H%M').ge('1500').to_numpy();early=g.ts.dt.strftime('%H%M').le('0930').to_numpy()
        if v.sum()<=0 or not late.any():continue
        lr=np.log(p[-1]/p[np.flatnonzero(late)[0]])
        er=np.log(p[np.flatnonzero(early)[-1]]/p[0])
        rows.append(dict(date=date,minute_close=p[-1],minute_count=len(g),rv=float(r@r),down_variation=float(np.sum(np.minimum(r,0)**2)),path_efficiency=float(abs(np.log(p[-1]/p[0]))/(np.abs(r).sum()+1e-12)),ret_autocorr=pd.Series(r).autocorr(),late_return=lr,early_return=er,late_volume_share=float(v[late].sum()/v.sum()),vwap_gap=float(p[-1]/np.average(p,weights=v)-1),signed_volume_proxy=float(np.dot(np.sign(np.r_[0,r]),v)/v.sum()),late_vs_early=lr-er,rv_ratio=float(np.sum(r[-30:]**2)/(r@r+1e-12))))
    return pd.DataFrame(rows)

def dataset(root,out):
    bars=pd.read_parquet(root/'silver/bars.parquet')
    bars=bars[bars.symbol.isin(SYMBOLS)].copy()
    # Historical reconstruction, not a claim of original publication vintages.
    bars['available_at']=(bars.date+pd.Timedelta(hours=18)).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    f=enhance(extend(features(bars,Lake(root).calendar())))
    market=pd.read_parquet(root/'silver/market.parquet')
    f['market_range']=f.date.map(market.drop_duplicates('date').set_index('date').range_pct)
    f=f.merge(build(f,bars,'standard'),on=['symbol','date'],validate='one_to_one')
    parts=[];audit=[]
    for symbol in SYMBOLS:
        path=root/f'runs/minute-depth-20261017-161326/{symbol}-1.parquet'
        raw=pd.read_parquet(path);m=minute_features(raw);m['symbol']=symbol
        m=m.merge(bars[bars.symbol.eq(symbol)][['date','close']],on='date',validate='one_to_one')
        match=m.minute_close.eq(m.close)
        audit.append(dict(symbol=symbol,source_rows=len(raw),session_days=len(m),matching_close_days=int(match.sum()),excluded_close_mismatch=int((~match).sum()),source_hash=digest(path.read_bytes())))
        parts.append(m[match].drop(columns='close'))
    f=f.merge(pd.concat(parts),on=['symbol','date'],validate='one_to_one')
    # Verified reference continuity in features() prevents corporate-action returns.
    f=f[f.direction_return.notna() & f.direction_return.ne(0) & f.return_1.notna() & f.target_date.notna()].copy()
    f=f.sort_values(['target_date','symbol']).reset_index(drop=True)
    f.to_parquet(out/'features.parquet',index=False);atomic_json(out/'data-audit.json',audit)
    return f

def models():
    return dict(logistic=LogisticRegression(C=.1,max_iter=1000),lightgbm=LGBMClassifier(n_estimators=120,num_leaves=5,min_child_samples=35,verbosity=-1,n_jobs=2,random_state=17),catboost=CatBoostClassifier(iterations=120,depth=3,learning_rate=.03,verbose=False,thread_count=2,random_seed=17,allow_writing_files=False),random_forest=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=20,n_jobs=2,random_state=17),extra_trees=ExtraTreesClassifier(n_estimators=200,max_depth=4,min_samples_leaf=20,n_jobs=2,random_state=17),hist_gradient=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=10,random_state=17),rbf_svm=SVC(C=.2,probability=True,random_state=17),gaussian_nb=GaussianNB(var_smoothing=.1),mlp=MLPClassifier(hidden_layer_sizes=(16,8),alpha=10,max_iter=600,random_state=17))

def calibrator(p,y):
    x=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
    return LogisticRegression(C=1,max_iter=1000).fit(x,y)

def apply_cal(c,p):
    if c is None:return np.clip(p,1e-6,1-1e-6)
    x=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
    return c.predict_proba(x)[:,1]

def report_metrics(f,p,prior):
    r=evaluate(f,p,prior)
    r.update(accuracy=float(accuracy_score(f.label,p>=.5)),roc_auc=float(roc_auc_score(f.label,p)),higher_probability_direction_accuracy=float(np.mean(np.where(p>.5,f.label,1-f.label)[p!=.5])) if np.any(p!=.5) else None)
    confidence=np.maximum(p,1-p);hit=np.where(p>=.5,f.label,1-f.label)
    r['confidence_order']=[dict(low=lo,high=hi,n=int(((confidence>=lo)&(confidence<hi)).sum()),predicted=float(confidence[(confidence>=lo)&(confidence<hi)].mean()) if ((confidence>=lo)&(confidence<hi)).any() else None,actual=float(hit[(confidence>=lo)&(confidence<hi)].mean()) if ((confidence>=lo)&(confidence<hi)).any() else None) for lo,hi in [(.5,.55),(.55,.6),(.6,.65),(.65,.7),(.7,1.000001)]]
    return r

def run():
    root=local_settings().data_dir;out=root/'runs'/RUN;out.mkdir(exist_ok=True)
    protected={str(p):digest(p.read_bytes()) for d in ['config','forward','direction-forward'] for p in (root/d).rglob('*') if p.is_file()}
    protected.update({str(p):digest(p.read_bytes()) for p in root.rglob('*.joblib') if out not in p.parents})
    protocol=dict(created=utcnow(),target='close t -> next KRX session close; conditional on nonflat',train='target < 2026-04-01',calibration='2026-04-01 <= target < 2026-06-01',selection='2026-06-01 <= target < 2026-07-01',evaluation='2026-07-01 onward',selection_rule='lowest date-balanced selection Brier then Log Loss, both directions symmetric; no test selection',promotion='none unless independent evidence later; existing models preserved',limitations=['3 large-cap stocks only','2025/2026 overlap previous range test; not globally unseen','Chronos pretraining corpus contamination unverified','Historical revisions/publication timestamps not certified','Signed-volume feature is a tick-rule proxy, NOT observed order flow','Free investor-flow dataset ends 2024, so cannot be silently zero-filled in 2026'],code_hash=digest(Path(__file__).read_bytes()))
    if (out/'report.json').exists():raise RuntimeError('Completed run is immutable')
    atomic_json(out/'protocol.json',protocol)
    f=dataset(root,out);print('DATA',len(f),f.target_date.min(),f.target_date.max(),flush=True)
    parts=dict(train=f[f.target_date.lt('2026-04-01')],cal=f[f.target_date.ge('2026-04-01')&f.target_date.lt('2026-06-01')],selection=f[f.target_date.ge('2026-06-01')&f.target_date.lt('2026-07-01')],test=f[f.target_date.ge('2026-07-01')])
    for name,g in parts.items():
        if g.empty or g.label.nunique()<2:raise ValueError('Insufficient split '+name)
    atomic_json(out/'splits.json',{k:dict(n=len(g),dates=g.target_date.nunique(),first=str(g.target_date.min()),last=str(g.target_date.max())) for k,g in parts.items()})
    tech=sorted(set(COLS+sum(GROUPS.values(),[])))
    groups=dict(base=COLS,technical=tech,technical_minute=tech+MINUTE)
    prior=float(parts['train'].label.mean());preds={};selection={};errors={}
    with threadpool_limits(limits=2):
        for group,cols in groups.items():
            for name,est in models().items():
                key=group+'__'+name
                try:
                    m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),est)
                    m.fit(parts['train'][cols],parts['train'].label)
                    raw={k:m.predict_proba(g[cols])[:,1] for k,g in parts.items() if k!='train'}
                    c=calibrator(raw['cal'],parts['cal'].label)
                    for method,cm in [('raw',None),('sigmoid',c)]:
                        k=key+'__'+method;preds[k]={p:apply_cal(cm,v) for p,v in raw.items()}
                        b,ll=losses(parts['selection'].label,preds[k]['selection']);d=pd.DataFrame(dict(date=parts['selection'].target_date,b=b,ll=ll)).groupby('date').mean()
                        selection[k]=dict(brier=float(d.b.mean()),log_loss=float(d.ll.mean()))
                    joblib.dump(dict(model=m,calibrator=c,columns=cols),out/(key+'.joblib'))
                    print('FIT',key,flush=True)
                except Exception as e:errors[key]=type(e).__name__+': '+str(e);print('ERROR',key,errors[key],flush=True)
    # Optional Chronos-2 predictions are produced separately on exactly these rows.
    cp=out/'chronos-predictions.parquet'
    if cp.exists():
        ch=pd.read_parquet(cp).set_index(['symbol','target_date'])
        raw={k:ch.loc[pd.MultiIndex.from_frame(g[['symbol','target_date']]),'p_up'].to_numpy() for k,g in parts.items() if k!='train'}
        c=calibrator(raw['cal'],parts['cal'].label)
        for method,cm in [('raw',None),('sigmoid',c)]:
            k='chronos2__'+method;preds[k]={p:apply_cal(cm,v) for p,v in raw.items()};b,ll=losses(parts['selection'].label,preds[k]['selection']);selection[k]=dict(brier=float(b.mean()),log_loss=float(ll.mean()))
    chosen=min(selection,key=lambda k:(selection[k]['brier'],selection[k]['log_loss']))
    atomic_json(out/'selection.json',dict(chosen=chosen,scores=selection))
    results={};test=parts['test'];output=test[['symbol','date','target_date','label']].copy()
    from .direction_calibration_study import calibrated
    # Only initial COLS have identical market-feature semantics here. The recent
    # model requires full-universe ranks, not three-stock cross-sectional ranks.
    for old in ['initial_refit__sigmoid']:
        saved=joblib.load(root/f'runs/direction-calibration-study-20261017/{old}.joblib')
        raw=saved['model'].predict_proba(test[saved['cols']])[:,1]
        p=calibrated(saved['calibrator'],test,raw,saved['method'])
        preds['frozen_'+old]={'test':p}
    for key,p in preds.items():
        output[key]=p['test'];results[key]=report_metrics(test,p['test'],prior)
    output['training_prior']=prior;results['training_prior']=report_metrics(test,np.full(len(test),prior),prior)
    reference='base__logistic__sigmoid'
    contrasts={k:paired(test,[(-1,preds[reference]['test']),(1,p['test'])]) for k,p in preds.items()}
    output.to_parquet(out/'test-predictions.parquet',index=False)
    unchanged=all(digest(Path(p).read_bytes())==v for p,v in protected.items())
    result=dict(protocol=protocol,selected=chosen,reference=reference,results=results,contrasts=contrasts,errors=errors,production_unchanged=unchanged,artifact_dir=str(out),chronos_included=cp.exists(),new_fitted_estimators=len(list(out.glob('*.joblib'))))
    atomic_json(out/'report.json',result);atomic_json(root/'reports/intraday-probability-refit.json',result)
    print('DONE',chosen,'unchanged',unchanged,flush=True)

if __name__=='__main__':run()
