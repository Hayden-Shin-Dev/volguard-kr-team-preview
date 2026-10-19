"""Isolated pre-close flow snapshots and prospective research issuance; no release."""
import argparse,json,warnings
from datetime import datetime
from dataclasses import replace
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .config import KST
from .data import Lake
from .direction_contract import window,aware,record
from .direction_forward import fetch_day,evaluate,Unavailable
from .long_history_probability import features,apply_cal
from .state_weighted_study import state_features,STATE
from .flow_ablation import flow_features,RUN,DYNAMICS
from .new_information_sources import Reader
from .platform_study import local_settings
from .util import atomic_json,digest,utcnow,exclusive_run

def snapshot(settings):
    root=settings.data_dir;out=root/'flow-forward';out.mkdir(exist_ok=True)
    (out/'kiwoom-spec.json').write_bytes((root/'runs'/RUN/'kiwoom-spec.json').read_bytes())
    symbols=json.loads((root/'runs'/RUN/'universe.json').read_text())['symbols']
    now=datetime.now(KST);day=now.strftime('%Y%m%d');saved=[]
    acquisition=out/'acquisitions'/now.strftime('%Y%m%dT%H%M%S%f');acquisition.mkdir(parents=True)
    (acquisition/'kiwoom-spec.json').write_bytes((out/'kiwoom-spec.json').read_bytes())
    reader=Reader(acquisition)
    try:
        for symbol in symbols:
            v=reader.get('ka10059',dict(dt=day,stk_cd=symbol,amt_qty_tp='1',trde_tp='0',unit_tp='1'))
            # Reader caches the actual receipt time. Never relabel an old response as freshly received.
            payload=dict(symbol=symbol,received_at=v['retrieved_at'],rows=v['response'].get('stk_invsr_orgn',[]),exchange='KRX')
            dest=out/'snapshots'/symbol/(digest(payload)+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():atomic_json(dest,payload)
            saved.append(symbol)
    finally:reader.client.close()
    atomic_json(out/'snapshot-status.json',dict(at=utcnow(),symbols=len(saved),information='actual receipt timestamp; historical revisions retained by distinct payload hash'))
    return len(saved)

def select_snapshot(folder,cutoff):
    choices=[]
    for path in folder.glob('*.json'):
        data=json.loads(path.read_text());stamp=aware(data['received_at'])
        if stamp<=aware(cutoff):choices.append((stamp,path,data))
    if not choices:raise Unavailable('장 마감 전에 수신·보존한 수급 응답 없음: 15:00 수집 작업 확인')
    return max(choices,key=lambda x:x[0])[1:]

def price_frame(settings,day,current,source):
    root=settings.data_dir
    raw=pd.read_parquet(root/'runs/long-history-foundation-20261017/normalized-bars.parquet')
    universe=set(raw.symbol)
    current=current[current.symbol.isin(universe)].copy()
    # Include all subsequent official snapshots, without rewriting the historical training source.
    for p in sorted((root/'flow-forward/prices').glob('*.parquet')):
        add=pd.read_parquet(p);raw=pd.concat([raw,add],ignore_index=True)
    raw=raw[raw.date.lt(pd.Timestamp(day))&raw.symbol.isin(universe)]
    add=current[['symbol','date','open','high','low','close','volume','turnover']].copy()
    add['valid']=current.valid_label.to_numpy()
    raw=pd.concat([raw,add],ignore_index=True).drop_duplicates(['symbol','date'],keep='last')
    price_dir=root/'flow-forward/prices';price_dir.mkdir(exist_ok=True)
    dest=price_dir/(source['hash']+'.parquet')
    if not dest.exists():add.to_parquet(dest,index=False)
    calendar=sorted(set(raw.date)|{pd.Timestamp(s['date']) for s in Lake(root).calendar() if s['date']<=day})
    # All state statistics use the same fixed 201-symbol source universe as training.
    tail=pd.Timestamp(day)-pd.Timedelta(days=180)
    raw=raw[raw.date.ge(tail)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',FutureWarning)
        f=features(raw,pd.DatetimeIndex([d for d in calendar if d>=tail]));state=state_features(raw)[['symbol','date']+STATE]
    f=f.merge(state,on=['symbol','date'],validate='one_to_one')
    f=f[f.date.eq(pd.Timestamp(day))&f.usable].copy();f['market_range']=source['market_range']
    return f,calendar

def install(settings):
    root=settings.data_dir;study=root/'runs'/RUN;out=root/'flow-forward';out.mkdir(exist_ok=True)
    selection=json.loads((study/'selection.json').read_text());name=selection['chosen'];src=study/(name+'.joblib')
    h=digest(src.read_bytes());dest=out/(h+'.joblib')
    if not dest.exists():dest.write_bytes(src.read_bytes())
    config=dict(model=name,hash=h,path=str(dest),research_only=True,automatic_promotion=False,selection='2019-2020 only; retrospective review does not select version',installed_at=utcnow())
    path=out/'config.json'
    if path.exists() and json.loads(path.read_text())['hash']!=h:raise ValueError('Existing forward candidate must be versioned explicitly')
    if not path.exists():atomic_json(path,config)
    market=study/'market-components'
    if (market/'selection.json').exists():
        name=json.loads((market/'selection.json').read_text())['selected'];src=market/(name+'.joblib');h=digest(src.read_bytes());dest=out/(h+'.joblib')
        if not dest.exists():dest.write_bytes(src.read_bytes())
        mp=out/'market-config.json'
        mc=dict(model=name,hash=h,path=str(dest),research_only=True,selection='2019-2020 only',automatic_promotion=False)
        if mp.exists() and json.loads(mp.read_text())['hash']!=h:raise ValueError('Market research version already frozen')
        if not mp.exists():atomic_json(mp,mc)
    return config

def issue(settings,day,preview=False):
    root=settings.data_dir;out=root/'flow-forward';now=datetime.now(KST)
    calendar={'sessions':Lake(root).calendar()};dates=window(calendar,day)
    if not preview and (now.date().isoformat()!=day or now<aware(dates['feature_cutoff'])):raise Unavailable('당일 장 마감 후에만 발행: 소급 발행 금지')
    conf=json.loads((out/'config.json').read_text());path=Path(conf['path'])
    if digest(path.read_bytes())!=conf['hash']:raise Unavailable('연구 모델 해시 변경')
    model=joblib.load(path);current,source=fetch_day(settings,day);f,days=price_frame(settings,day,current,source)
    symbols=json.loads((root/'runs'/RUN/'universe.json').read_text())['symbols'];results=[]
    snapshots={};flowrows=[]
    for symbol in symbols:
        try:
            sp,snap=select_snapshot(out/'snapshots'/symbol,dates['feature_cutoff'])
            ff=flow_features(pd.DataFrame(snap['rows']),days);snapshots[symbol]=(sp,snap,ff)
            complete=ff.dropna();flowrows.append(complete)
        except Unavailable:pass
    marketcols=DYNAMICS('foreign')+DYNAMICS('institution')
    marketrow=None
    if flowrows:
        flows=pd.concat(flowrows);market=flows.groupby('date')[marketcols].mean();counts=flows.groupby('date').size()
        if pd.Timestamp(day) in market.index and counts.loc[pd.Timestamp(day)]>=20:
            marketrow={'market_'+k:float(v) for k,v in market.loc[pd.Timestamp(day)].items()}
    configs=[(conf,model,False)]
    if (out/'champion-config.json').exists():
        cc=json.loads((out/'champion-config.json').read_text());cp=Path(cc['path'])
        if digest(cp.read_bytes())!=cc['hash']:raise Unavailable('비교 기준모델 해시 변경')
        configs.insert(0,(cc,joblib.load(cp),False))
    if (out/'market-config.json').exists():
        mc=json.loads((out/'market-config.json').read_text());mp=Path(mc['path'])
        if digest(mp.read_bytes())!=mc['hash']:raise Unavailable('시장 수급 연구 모델 해시 변경')
        configs.append((mc,joblib.load(mp),True))
    for symbol in symbols:
      for conf,model,is_market in configs:
        try:
            is_champion=conf['model']=='champion_long_history'
            row=f[f.symbol.eq(symbol)].copy()
            if not is_champion:
                if symbol not in snapshots:raise Unavailable('장 마감 전에 수신·보존한 수급 응답 없음: 15:00 수집 작업 확인')
                sp,snap,flows=snapshots[symbol]
                row=row.merge(flows,on='date',validate='one_to_one')
                row['foreign_trend']=row.foreign_intensity5*row.market_trend;row['institution_trend']=row.institution_intensity5*row.market_trend
                row['flow_agreement']=row.foreign_intensity5*row.institution_intensity5
            if is_market:
                if marketrow is None:raise Unavailable('마감 전 유효 시장 수급 20종목 미만')
                for k,v in marketrow.items():row[k]=v
                row['foreign_trend']=row.market_foreign_intensity5*row.market_trend;row['institution_trend']=row.market_institution_intensity5*row.market_trend
            if len(row)!=1 or row[model['columns']].isna().any(axis=None):raise Unavailable('전 거래일 수급 또는 20일 연속 유효 자료 부족')
            p=float(apply_cal(model['calibration'],model['model'].predict_proba(row[model['columns']])[:,1])[0])
            anchor=current[current.symbol.eq(symbol)]
            evidence=[dict(kind='krx_regular_daily',session='regular',source_hash=source['hash'],observed_at=dates['feature_cutoff'],available_at=source['received_at'])]
            if not is_champion:evidence.append(dict(kind='investor_flow_snapshot',source_hash=digest(sp.read_bytes()),observed_at=str(row.observed_date.iloc[0].date())+'T23:59:59+09:00',available_at=snap['received_at'],time_semantics='conservative end-of-observation-day bound'))
            if is_market:
                evidence += [dict(kind='market_flow_component',source_hash=digest(spp.read_bytes()),observed_at=str(ff.loc[ff.date.eq(pd.Timestamp(day)),'observed_date'].iloc[0].date())+'T23:59:59+09:00',available_at=ss['received_at'],time_semantics='conservative end-of-observation-day bound') for spp,ss,ff in snapshots.values() if len(ff.loc[ff.date.eq(pd.Timestamp(day))].dropna())]
            if not preview:
                record(out,calendar,day,symbol,'flow-'+conf['model']+'-'+conf['hash'][:12],p,evidence,details=dict(anchor_close=float(anchor.close.iloc[0]),model_hash=conf['hash'],research_only=True,features=row[model['columns']].iloc[0].to_dict()))
            results.append(dict(symbol=symbol,model=conf['model'],status='preview_not_issued' if preview else 'issued',p_up=p,p_down=1-p))
        except FileExistsError:results.append(dict(symbol=symbol,model=conf['model'],status='already_issued'))
        except (ValueError,KeyError) as exc:results.append(dict(symbol=symbol,model=conf['model'],status='unavailable',reason=str(exc) if isinstance(exc,Unavailable) else type(exc).__name__))
    result=dict(at=utcnow(),day=day,target=dates['target_date'],results=results,preview=preview)
    atomic_json(out/('preview.json' if preview else 'batch-status.json'),result);return result

def run(settings,day):
    out=settings.data_dir/'flow-forward'
    if not (out/'config.json').exists():return
    with exclusive_run(out/'run.lock'):
        evaluate(replace(settings,data_dir=out))
        try:
            return issue(settings,day)
        finally:
            summarize(settings)

def summarize(settings):
    """Only score actual immutable issues, never replay predictions as forward results."""
    from .probability_quality import evaluate as evaluate_probabilities
    from sklearn.metrics import roc_auc_score
    from .institution_components import EDGES
    from .probability_quality import band
    from .direction_historical_reliability import draws
    from .direction_scope_control import paired
    root=settings.data_dir/'flow-forward';rows=[];issued=0
    for p in (root/'direction-forward').glob('????-??-??/*.json'):
        entry=json.loads(p.read_text());issued+=1
        op=root/'direction-forward/outcomes'/entry['target_date']/p.name
        if not op.exists():continue
        result=json.loads(op.read_text())
        if result.get('label') not in [0,1]:continue
        if result.get('prediction_hash')!=digest(p.read_bytes()):continue
        rows.append(dict(model=entry['model_version'],symbol=entry['symbol'],target_date=entry['target_date'],label=result['label'],p=entry['p_up']))
    f=pd.DataFrame(rows);models={};comparisons={}
    if not f.empty:
        for name,g in f.groupby('model'):
            models[name]=evaluate_probabilities(g.reset_index(drop=True),g.p.to_numpy(),.5)
            models[name]['roc_auc']=float(roc_auc_score(g.label,g.p)) if g.label.nunique()==2 else None
            dates=sorted(g.target_date.unique());ix=draws(dates)
            models[name]['all_probability_bands']=[band(g,g.p.to_numpy(),side,lo,hi,ix,dates) for side in ['up','down'] for lo,hi in EDGES]
        champions=[n for n in models if 'champion_long_history' in n]
        if len(champions)==1:
            champion=f[f.model.eq(champions[0])]
            for n in models:
                if n==champions[0]:continue
                shared=f[f.model.eq(n)].merge(champion[['symbol','target_date','label','p']],on=['symbol','target_date','label'],validate='one_to_one',suffixes=('','_champion'))
                comparisons[n]=dict(rows=len(shared),dates=int(shared.target_date.nunique()),loss_difference=paired(shared,[(-1,shared.p_champion.to_numpy()),(1,shared.p.to_numpy())]) if shared.target_date.nunique()>=40 else None,status='research_only_no_automatic_promotion')
    atomic_json(root/'reliability.json',dict(at=utcnow(),issued=issued,scored_nonflat=len(rows),models=models,champion_challenger=comparisons,benchmark='fixed neutral 50% for BSS; frozen full-history Champion for matched contrasts',research_only=True,automatic_promotion=False))
    return models

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['snapshot','install','run','preview']);p.add_argument('--date',default=datetime.now(KST).date().isoformat());a=p.parse_args();s=local_settings()
    result=snapshot(s) if a.command=='snapshot' else install(s) if a.command=='install' else issue(s,a.date,preview=True) if a.command=='preview' else run(s,a.date)
    print(json.dumps(result,ensure_ascii=False))
