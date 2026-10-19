"""Full KRX/KOSDAQ snapshot for the fixed daily research universe, no issuance."""
from datetime import datetime
import numpy as np
import pandas as pd
from .config import KST
from .providers import KRX, request_json, ProviderError
from .util import atomic_json, digest

def normalize_markets(payloads, day, symbols):
    frames=[]
    for market in ['kospi','kosdaq']:
        payload=payloads[market]
        raw=pd.DataFrame(payload.get('OutBlock_1',[]))
        if raw.empty:raise ProviderError('RESEARCH_OFFICIAL_BAR_EMPTY:'+market+':'+day)
        if 'BAS_DD' not in raw or 'CMPPREVDD_PRC' not in raw:
            raise ProviderError('RESEARCH_OFFICIAL_DATE_OR_REFERENCE_MISSING')
        frame=KRX.normalize(payload,day)
        frame['date']=pd.to_datetime(frame.date)
        frame['change']=pd.to_numeric(frame.CMPPREVDD_PRC.astype(str).str.replace(',',''),errors='coerce')
        frame['source_market']=market;frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    frame=frame[frame.symbol.isin(symbols)].copy()
    if frame.duplicated(['symbol','date']).any():raise ProviderError('RESEARCH_DUPLICATE_MARKET_LISTING')
    missing=sorted(set(symbols)-set(frame.symbol))
    if missing:raise ProviderError('RESEARCH_FIXED_UNIVERSE_MISSING:'+','.join(missing))
    if not np.isfinite(frame[['open','high','low','close','volume','turnover','change']]).all().all():
        raise ProviderError('RESEARCH_NONFINITE_OFFICIAL_BAR')
    # Zero-volume/suspended rows remain present for build_features to reject explicitly.
    return frame[['symbol','date','open','high','low','close','volume','turnover','change','source_market']].sort_values('symbol').reset_index(drop=True)

def fetch_snapshot(root, day, symbols):
    client=KRX();payloads={}
    for market,endpoint in [('kospi','sto/stk_bydd_trd'),('kosdaq','sto/ksq_bydd_trd')]:
        payloads[market]=request_json('GET',f'{client.BASE}/{endpoint}',headers={'AUTH_KEY':client.key},params={'basDd':pd.Timestamp(day).strftime('%Y%m%d')})
    received=datetime.now(KST).isoformat()
    archive=dict(day=day,received_at=received,payloads=payloads)
    sha=digest(archive);path=root/'sources'/(sha+'.json')
    if not path.exists():atomic_json(path,archive)
    frame=normalize_markets(payloads,day,symbols)
    return frame,dict(source_hash=sha,received_at=received,path=str(path))
