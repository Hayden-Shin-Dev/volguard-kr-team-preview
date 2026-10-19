"""Frozen daily-sequence research inference; no issuance or production selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .util import digest
from .long_history_probability import apply_cal

COLUMNS=['return_1','gap','body_pct','close_location','volume_ratio','turnover_ratio','market_trend','market_vol','breadth','relative20']

def make_windows(frame, forecast_date, calendar):
    """Caller supplies audited, available features and authoritative trading calendar."""
    day=pd.Timestamp(forecast_date).normalize();dates=pd.DatetimeIndex(calendar)
    if dates.has_duplicates or not dates.is_monotonic_increasing or day not in dates:
        raise ValueError('Invalid trading calendar or forecast date')
    f=frame.copy();f['date']=pd.to_datetime(f.date)
    if f.date.gt(day).any():raise ValueError('Future feature rows supplied')
    if f.duplicated(['symbol','date']).any():raise ValueError('Duplicate symbol/date')
    expected=dates[dates<=day][-20:];symbols=[];arrays=[];rejected={}
    for symbol,g in f.groupby('symbol',sort=True):
        g=g.sort_values('date').tail(20)
        if len(expected)!=20 or not pd.DatetimeIndex(g.date).equals(expected):
            rejected[symbol]='20 consecutive trading sessions unavailable';continue
        if not (g.price_valid.eq(True)&g.continuity_verified.eq(True)).all():
            rejected[symbol]='Unverified price or corporate-action continuity';continue
        values=g[COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():rejected[symbol]='Nonfinite feature';continue
        symbols.append(symbol);arrays.append(values)
    return symbols,np.stack(arrays) if arrays else np.empty((0,20,len(COLUMNS))),rejected

class FrozenDailyCandidate:
    def __init__(self, manifest):
        import torch
        from torch import nn
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=nn.GRU(10,24,batch_first=True);self.head=nn.Linear(24,1);self.score=nn.Linear(24,1,bias=False)
            def forward(self,x):return self.head(self.encoder(x)[0][:,-1]).squeeze(-1)
        self.spec=json.loads(Path(manifest).read_text(encoding='utf8'));self.version=digest(self.spec)
        if not self.spec.get('research_only') or self.spec.get('automatic_promotion'):
            raise ValueError('Research-only manifest required')
        if len(self.spec['models'])!=3:raise ValueError('Exactly three pinned members required')
        self.models=[]
        for entry in self.spec['models']:
            weight=Path(entry['checkpoint']);transform=Path(entry['transform'])
            if digest(weight.read_bytes())!=entry['checkpoint_sha256'] or digest(transform.read_bytes())!=entry['transform_sha256']:
                raise ValueError('Frozen candidate artifact hash mismatch')
            t=json.loads(transform.read_text());mean=np.array(t['mean']);scale=np.array(t['scale'])
            if t['columns']!=COLUMNS or mean.shape!=(10,) or scale.shape!=(10,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
                raise ValueError('Invalid feature transform')
            model=Encoder();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));model.eval()
            self.models.append((model,mean,scale,t['calibration']))
    def predict(self, windows):
        import torch
        a=np.asarray(windows,dtype=float)
        if a.ndim!=3 or a.shape[1:]!=(20,10) or not np.isfinite(a).all():raise ValueError('Expected finite N by20 by10 audited inputs')
        if not len(a):return np.empty(0)
        members=[]
        with torch.no_grad():
            for model,mean,scale,calibration in self.models:
                x=torch.tensor(((a-mean)/scale).astype('float32'))
                raw=np.concatenate([torch.sigmoid(model(chunk)).numpy() for chunk in x.split(512)])
                members.append(apply_cal(calibration,raw))
        return np.mean(members,axis=0)
