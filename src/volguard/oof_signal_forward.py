"""Failure-isolated prospective OOF research. Never changes a production model."""
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from .config import KST
from .data import Lake
from .direction import COLS
from .direction_data_audit import build_features
from .direction_contract import window,aware,record
from .direction_forward import fetch_day,evaluate,Unavailable
from .oof_signal import enrich,up_probability,combined_probability,EXTRA
from .long_history_probability import apply_cal
from .store import Store
from .util import digest,atomic_json,exclusive_run,utcnow

STUDY='direction-oof-signal-20261017'
AUDIT='direction-pipeline-audit-kosdaq-20261017'

def code_hash():
    names=['oof_signal.py','oof_signal_forward.py','direction_data_audit.py','direction_contract.py','direction_forward.py','data.py']
    return digest({n:digest((Path(__file__).parent/n).read_bytes()) for n in names})

def install(settings,refresh_unissued=False):
    root=settings.data_dir;out=root/'oof-signal-forward';out.mkdir(exist_ok=True)
    selection=json.loads((root/'runs'/STUDY/'selection.json').read_text())
    challenger=min((n for n in selection['scores'] if n!='baseline'),key=lambda n:(selection['scores'][n]['brier'],selection['scores'][n]['log_loss']))
    specs=[]
    for name in ['baseline',challenger]:
        src=root/'runs'/STUDY/(name+'.joblib');h=digest(src.read_bytes());dest=out/(h+'.joblib')
        if not dest.exists():dest.write_bytes(src.read_bytes())
        specs.append({'name':name,'path':str(dest),'hash':h,'version':'oof-research-'+h[:16]})
    history=root/'runs'/AUDIT/'official-bars.parquet'
    train=pd.read_parquet(root/'runs'/STUDY/'features.parquet',columns=['target_date','label'])
    prior=float(train.loc[train.target_date.lt('2021-01-01'),'label'].mean())
    config={'models':specs,'prior_up':prior,'history_path':str(history),'history_hash':digest(history.read_bytes()),'code_hash':code_hash(),'created_at':utcnow(),'research_only':True,'automatic_promotion':False,'selection':'2016/2018/2020 development only; challenger failed superiority, retained for research comparison only'}
    p=out/'config.json'
    if p.exists():
        old=json.loads(p.read_text())
        if old['models']!=specs or old['code_hash']!=config['code_hash']:
            if not refresh_unissued or list((out/'direction-forward').glob('????-??-??/*.json')):raise ValueError('Version already pinned')
            atomic_json(out/'config-versions'/(digest(old)+'.json'),old)
        else:return old
    atomic_json(p,config);return config

def prepare(settings,day,current,source,config):
    out=settings.data_dir/'oof-signal-forward';path=Path(config['history_path'])
    if digest(path.read_bytes())!=config['history_hash']:raise Unavailable('고정 연구 원본 해시 변경')
    history=pd.read_parquet(path);universe=set(history.symbol.unique())
    for p in sorted((out/'prices').glob('*.parquet')):
        history=pd.concat([history,pd.read_parquet(p)],ignore_index=True)
    current=current[current.symbol.isin(universe)].copy()
    if universe-set(current.symbol):raise Unavailable('고정 201종목 공식 시장 단면 누락')
    current['change']=pd.to_numeric(current.CMPPREVDD_PRC.astype(str).str.replace(',',''),errors='coerce')
    cols=['symbol','date','open','high','low','close','volume','turnover','change']
    add=current[cols];tail=pd.Timestamp(day)-pd.Timedelta(days=200)
    history=history[history.date.lt(pd.Timestamp(day))&history.date.ge(tail)]
    bars=pd.concat([history[cols],add],ignore_index=True).drop_duplicates(['symbol','date'],keep='last')
    dates=pd.to_datetime([s['date'] for s in Lake(settings.data_dir).calendar() if tail<=pd.Timestamp(s['date'])<=pd.Timestamp(day)])
    f=enrich(build_features(bars,dates));f=f[f.date.eq(pd.Timestamp(day))].copy();f['market_range']=source['market_range']
    dest=out/'prices'/(source['hash']+'.parquet');dest.parent.mkdir(exist_ok=True)
    if not dest.exists():add.to_parquet(dest,index=False)
    return f

def issue(settings,day,preview=False):
    out=settings.data_dir/'oof-signal-forward';config=json.loads((out/'config.json').read_text());now=datetime.now(KST)
    calendar={'sessions':Lake(settings.data_dir).calendar()};dates=window(calendar,day)
    if not preview and (now.date().isoformat()!=day or now<aware(dates['feature_cutoff'])):raise Unavailable('당일 장 마감 후 발행만 가능: 소급 발행 금지')
    if config['code_hash']!=code_hash():raise Unavailable('연구 추론 코드 변경: 새 버전 필요')
    current,source=fetch_day(replace(settings,data_dir=out),day);f=prepare(settings,day,current,source,config)
    saved=out/'inputs'/(source['hash']+'.parquet');saved.parent.mkdir(exist_ok=True)
    if not saved.exists():f.to_parquet(saved,index=False)
    symbols=[x['symbol'] for x in Store(settings.db_path).watchlist()];results=[]
    for spec in config['models']:
        path=Path(spec['path'])
        if digest(path.read_bytes())!=spec['hash']:raise Unavailable('연구 모델 해시 변경')
        b=joblib.load(path)
        for symbol in symbols:
            g=f[f.symbol.eq(symbol)];r={'symbol':symbol,'model':spec['name']}
            if len(g)!=1 or not g[COLS+EXTRA].notna().all().all():
                results.append(dict(r,status='unavailable',reason='정규장 연속 가격 또는 피처 준비 구간 부족'));continue
            base=up_probability(b['foundation'],g,COLS);p=float(apply_cal(b['calibration'],combined_probability(base,b['detector'],g,b['kind'],b['strength']))[0])
            if preview:r.update(status='preview_not_issued',p_up=p,p_down=1-p)
            else:
                try:
                    record(out,calendar,day,symbol,spec['version'],p,[{'kind':'krx_regular_daily','session':'regular','source_hash':source['hash'],'observed_at':dates['feature_cutoff'],'available_at':source['received_at']}],details={'model':spec['name'],'model_hash':spec['hash'],'anchor_close':float(current[current.symbol.eq(symbol)].close.iloc[0]),'input_hash':digest(saved.read_bytes()),'history_hash':config['history_hash'],'code_hash':config['code_hash']})
                    r.update(status='issued',p_up=p,p_down=1-p)
                except FileExistsError:r['status']='already_issued'
            results.append(r)
    return {'anchor_date':day,'target_date':dates['target_date'],'research_only':True,'preview':preview,'results':results}

def reliability(settings):
    from .probability_quality import evaluate as metrics,band
    from .direction_historical_reliability import draws
    from .institution_components import EDGES
    out=settings.data_dir/'oof-signal-forward';config=json.loads((out/'config.json').read_text());results=[]
    for spec in config['models']:
        rows=[];issued=flat=0
        for p in (out/'direction-forward').glob('????-??-??/*.json'):
            x=json.loads(p.read_text())
            if x['model_version']!=spec['version']:continue
            issued+=1;dest=out/'direction-forward/outcomes'/x['target_date']/p.name
            if not dest.exists():continue
            y=json.loads(dest.read_text())
            if y.get('prediction_hash')!=digest(p.read_bytes()):continue
            if y.get('direction')=='flat':flat+=1;continue
            if y.get('direction') not in ['up','down']:continue
            rows.append({'symbol':x['symbol'],'target_date':pd.Timestamp(x['target_date']),'label':int(y['direction']=='up'),'p_up':x['p_up']})
        f=pd.DataFrame(rows);v=metrics(f,f.p_up.to_numpy() if len(f) else [],config['prior_up']);bands=[]
        if len(f):
            dates=sorted(f.target_date.unique());ix=draws(dates)
            bands=[band(f,f.p_up.to_numpy(),side,lo,hi,ix,dates) for side in ['up','down'] for lo,hi in EDGES]
        results.append({'model':spec['name'],'issued':issued,'flat':flat,'scored':len(f),'pending':issued-flat-len(f),'metrics':v,'all_bands':bands})
    result={'at':utcnow(),'models':results,'research_only':True,'automatic_promotion':False};atomic_json(out/'reliability.json',result);return result

def run(settings,day):
    out=settings.data_dir/'oof-signal-forward'
    if not (out/'config.json').exists():return
    with exclusive_run(out/'batch.lock'):
        try:evaluate(replace(settings,data_dir=out))
        except Exception as e:atomic_json(out/'score-status.json',{'at':utcnow(),'error':type(e).__name__})
        try:result=issue(settings,day)
        except Exception as e:result={'at':utcnow(),'status':'unavailable','reason':str(e) if isinstance(e,Unavailable) else type(e).__name__}
        atomic_json(out/'batch-status.json',result)
        reliability(settings)
