"""Isolated fixed daily-sequence research journal and next-session scoring."""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .config import KST
from .daily_sequence_sources import fetch_snapshot
from .daily_sequence_candidate import FrozenDailyCandidate
from .daily_sequence_journal import issue_prepared
from .direction_contract import aware, window
from .direction_data_audit import build_features
from .direction_forward import score_entry
from .util import atomic_json, digest, exclusive_run

FOLDER='daily-sequence-forward'
def code_hash():
    names=['daily_sequence_forward.py','daily_sequence_sources.py','daily_sequence_candidate.py','daily_sequence_journal.py','direction_data_audit.py','direction_contract.py','direction_forward.py']
    return digest({n:digest((Path(__file__).parent/n).read_bytes()) for n in names})

def install(settings):
    root=settings.data_dir;folder=root/FOLDER;path=folder/'config.json'
    if path.exists():return json.loads(path.read_text())
    manifest=root/'runs/direction-daily-subepoch-stability-20261018/frozen-research-candidate.json'
    FrozenDailyCandidate(manifest)  # validates all pinned checkpoint and transform hashes
    history=root/'runs/direction-pipeline-audit-kosdaq-20261017/official-bars.parquet'
    bars=pd.read_parquet(history,columns=['symbol','date'])
    now=datetime.now(KST)
    if bars.date.max().date()>now.date():raise ValueError('Future history cannot initialize research')
    config=dict(manifest=str(manifest),manifest_hash=digest(manifest.read_bytes()),history=str(history),history_hash=digest(history.read_bytes()),
        history_received_at=now.isoformat(),symbols=sorted(bars.symbol.unique()),code_hash=code_hash(),research_only=True,automatic_promotion=False)
    atomic_json(path,config);return config

def load(settings):
    folder=settings.data_dir/FOLDER;c=json.loads((folder/'config.json').read_text())
    if c['code_hash']!=code_hash():raise ValueError('Frozen research implementation changed')
    for field in ['manifest','history']:
        if digest(Path(c[field]).read_bytes())!=c[field+'_hash']:raise ValueError('Frozen research artifact changed')
    return folder,c

def prepare(settings,folder,c,day,current,source):
    start=pd.Timestamp(day)-pd.Timedelta(days=200)
    history=pd.read_parquet(c['history'])
    cols=['symbol','date','open','high','low','close','volume','turnover','change']
    history=history[history.date.ge(start)&history.date.lt(day)][cols]
    evidence=[dict(kind='krx_regular_daily',session='regular',source_hash=c['history_hash'],
        observed_at=day+'T00:00:00+09:00',available_at=c['history_received_at'])]
    for meta in sorted((folder/'prices').glob('*.json')):
        m=json.loads(meta.read_text())
        if not start<=pd.Timestamp(m['day'])<pd.Timestamp(day):continue
        p=folder/'prices'/m['file']
        if digest(p.read_bytes())!=m['file_hash']:raise ValueError('Archived research prices changed')
        add=pd.read_parquet(p)
        if not add.date.eq(pd.Timestamp(m['day'])).all():raise ValueError('Archived research date mismatch')
        # Do not silently choose between revised snapshots of an already saved date.
        if history.date.eq(pd.Timestamp(m['day'])).any():continue
        history=pd.concat([history,add[cols]],ignore_index=True)
        evidence.append(dict(kind='krx_regular_daily',session='regular',source_hash=m['source_hash'],observed_at=m['day']+'T23:59:59+09:00',available_at=m['received_at']))
    bars=pd.concat([history,current[cols]],ignore_index=True)
    if bars.duplicated(['symbol','date']).any():raise ValueError('Duplicate prepared research prices')
    calendar=json.loads((settings.data_dir/'config/calendar.json').read_text())
    days=pd.to_datetime([s['date'] for s in calendar['sessions'] if start<=pd.Timestamp(s['date'])<=pd.Timestamp(day)])
    missing=set(days)-set(bars.date)
    if missing:raise ValueError('Missing research trading sessions; fetch official history before issuing')
    features=build_features(bars,days)
    dates=window(calendar,day)
    evidence.append(dict(kind='krx_regular_daily',session='regular',source_hash=source['source_hash'],observed_at=dates['feature_cutoff'],available_at=source['received_at']))
    prices=folder/'prices';prices.mkdir(exist_ok=True)
    metadata=prices/(day+'.json')
    if not metadata.exists():
        file=source['source_hash']+'.parquet';current[cols].to_parquet(prices/file,index=False)
        atomic_json(metadata,dict(day=day,file=file,file_hash=digest((prices/file).read_bytes()),**source))
    return features,calendar,evidence

def issue(settings,day):
    folder,c=load(settings)
    now=datetime.now(KST);calendar=json.loads((settings.data_dir/'config/calendar.json').read_text());dates=window(calendar,day)
    if now.date().isoformat()!=day or now<aware(dates['feature_cutoff']):raise ValueError('Same-day after-close only; no backfill')
    current,source=fetch_snapshot(folder,day,c['symbols'])
    features,calendar,evidence=prepare(settings,folder,c,day,current,source)
    model=FrozenDailyCandidate(c['manifest'])
    return issue_prepared(folder,model,features,calendar,day,c['symbols'],current.set_index('symbol').close.to_dict(),evidence)

def score(settings):
    folder,c=load(settings);now=datetime.now(KST);pending=[];scored=0;cache={}
    for path in sorted((folder/'direction-forward').glob('????-??-??/*.json')):
        entry=json.loads(path.read_text());dest=folder/'direction-forward/outcomes'/entry['target_date']/path.name
        if dest.exists() or now<aware(entry['target_close']):continue
        try:
            day=entry['target_date']
            if day not in cache:cache[day]=fetch_snapshot(folder,day,c['symbols'])
            bars,source=cache[day];g=bars[bars.symbol.eq(entry['symbol'])]
            if len(g)!=1:raise ValueError('Target close unavailable')
            row=g.iloc[0];valid=bool(row.volume>0 and row.low>0 and row.low<=min(row.open,row.close)<=max(row.open,row.close)<=row.high)
            target=dict(date=day,close=row.close,CMPPREVDD_PRC=row.change,valid_label=valid,exchange='KRX')
            result=score_entry(entry,target,now)
            if not dest.exists():atomic_json(dest,dict(**result,target_date=day,scored_at=now.isoformat(),prediction_hash=digest(path.read_bytes()),source_hash=source['source_hash']))
            scored+=1
        except (ValueError,RuntimeError) as exc:pending.append(dict(symbol=entry['symbol'],target_date=entry['target_date'],reason=str(exc)))
    result=dict(scored=scored,pending=pending);atomic_json(folder/'scoring-status.json',result);return result

def run(settings,day):
    folder=settings.data_dir/FOLDER
    if not (folder/'config.json').exists():return None
    with exclusive_run(folder/'batch.lock'):
        result=dict(research_only=True,production_changed=False,at=datetime.now(KST).isoformat())
        try:result['score']=score(settings)
        except Exception as exc:result['score_error']=type(exc).__name__
        try:result['issue']=issue(settings,day)
        except (ValueError,RuntimeError) as exc:result['issue']=dict(status='unavailable',reason=str(exc))
        atomic_json(folder/'batch-status.json',result)
        return result

if __name__=='__main__':
    import argparse
    from .platform_study import local_settings
    p=argparse.ArgumentParser();p.add_argument('command',choices=['install','run']);p.add_argument('--date',default=datetime.now(KST).date().isoformat());a=p.parse_args()
    value=install(local_settings()) if a.command=='install' else run(local_settings(),a.date)
    print(json.dumps({'status':'installed','research_only':True} if a.command=='install' else value,ensure_ascii=False))
