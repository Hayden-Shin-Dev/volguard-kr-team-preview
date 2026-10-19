"""Verify derived ownership tables against immutable provider pages before training."""
import json
from pathlib import Path
import pandas as pd
from volguard.util import digest

def load_verified(root: Path, state: dict) -> pd.DataFrame:
    if not state.get('complete'):
        raise ValueError('Ownership collection incomplete')
    rows=[]
    for page in state['pages']:
        path=root/'raw'/page['file']
        if digest(path.read_bytes())!=page['hash']:
            raise ValueError('Ownership raw hash mismatch')
        payload=json.loads(path.read_text(encoding='utf-8'))
        if payload['symbol']!=state['symbol']:
            raise ValueError('Ownership symbol mismatch')
        rows.extend(payload['rows'])
    if len(rows)!=state.get('rows',0):
        raise ValueError('Ownership checkpoint count mismatch')
    if not rows:return pd.DataFrame()
    original=pd.DataFrame(rows)
    if original.dt.duplicated().any():
        raise ValueError('Duplicate ownership dates')
    dates=pd.to_datetime(original.dt,format='%Y%m%d',errors='raise')
    if not dates.is_monotonic_decreasing:
        raise ValueError('Ownership pagination out of order')
    stored=pd.read_parquet(root/(state['symbol']+'.parquet'))
    pd.testing.assert_frame_equal(stored,original)
    if original.dt.min()!=state['first'] or original.dt.max()!=state['last']:
        raise ValueError('Ownership date bounds mismatch')
    return stored
