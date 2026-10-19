"""Reconstruct credit source rows from immutable pages before feature building."""
import json
from datetime import datetime
import pandas as pd
from volguard.util import digest

def load_verified(root,state):
    if not state.get('complete') or state.get('api')!='ka10013':
        raise ValueError('Credit source incomplete or wrong API')
    rows=[];invalid=0
    for page in state['pages']:
        p=root/'raw'/page['file']
        if digest(p.read_bytes())!=page['hash']:raise ValueError('Credit raw hash mismatch')
        payload=json.loads(p.read_text())
        if payload['symbol']!=state['symbol'] or payload['api']!='ka10013' or payload['request']!=state['request']:
            raise ValueError('Credit source identity mismatch')
        for row in payload['rows']:
            d=str(row.get('dt',''))
            try:
                assert len(d)==8 and d.isdigit()
                datetime.strptime(d,'%Y%m%d')
            except (AssertionError,ValueError):invalid+=1;continue
            if d>state['request']['dt']:raise ValueError('Credit date after request cutoff')
            rows.append(row)
    if len(rows)!=state.get('rows',0) or invalid!=state.get('invalid_date_rows',0):
        raise ValueError('Credit checkpoint count mismatch')
    if not rows:return pd.DataFrame()
    f=pd.DataFrame(rows)
    if f.dt.duplicated().any() or not f.dt.is_monotonic_decreasing:
        raise ValueError('Credit date pagination mismatch')
    if f.dt.min()!=state['first'] or f.dt.max()!=state['last']:raise ValueError('Credit date bounds mismatch')
    stored=pd.read_parquet(root/f"ka10013-{state['symbol']}.parquet")
    pd.testing.assert_frame_equal(f,stored)
    return f
