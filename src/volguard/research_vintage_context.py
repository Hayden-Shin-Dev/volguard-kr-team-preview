"""Strict historical-vintage parsing and backward-only research alignment."""
import io
import zipfile
import numpy as np
import pandas as pd

def observations(raw, series, vintage):
    day=pd.Timestamp(vintage).normalize()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(x.file_size for x in archive.infolist())>20_000_000:
                raise ValueError('Archive limit exceeded')
            frames=[pd.read_csv(io.BytesIO(archive.read(n))) for n in archive.namelist() if n.endswith('.csv')]
    else:
        frames=[pd.read_csv(io.BytesIO(raw))]
    column=series+'_'+day.strftime('%Y%m%d')
    matches=[f for f in frames if column in f]
    if len(matches)!=1:
        raise ValueError('Exact historical vintage column required')
    f=matches[0][['observation_date',column]].rename(columns={column:'value'}).copy()
    f.observation_date=pd.to_datetime(f.observation_date)
    f.value=pd.to_numeric(f.value,errors='coerce')
    f=f.dropna().sort_values('observation_date')
    if f.observation_date.duplicated().any() or (f.observation_date>day).any():
        raise ValueError('Duplicate or future observation')
    if not np.isfinite(f.value).all() or not f.value.gt(0).all():
        raise ValueError('Positive finite price/index required')
    return f

def align(days, source, columns, ttl=14):
    left=pd.DataFrame({'date':pd.to_datetime(days)}).sort_values('date')
    if left.date.duplicated().any():raise ValueError('Duplicate target day')
    left['cutoff']=(left.date+pd.Timedelta(hours=15,minutes=30)).dt.tz_localize('Asia/Seoul').dt.tz_convert('UTC')
    right=source.copy();right['available_at']=pd.to_datetime(right.available_at,utc=True)
    right['observed_date']=pd.to_datetime(right.observed_date)
    if right.available_at.duplicated().any():raise ValueError('Duplicate release timestamp')
    g=pd.merge_asof(left,right.sort_values('available_at'),left_on='cutoff',right_on='available_at',direction='backward')
    g['age_days']=(g.date-g.observed_date).dt.days
    g['usable']=g[columns].notna().all(axis=1)&g.age_days.between(0,ttl)
    if (g.available_at>g.cutoff).any():raise ValueError('Future source joined')
    g.loc[~g.usable,columns]=np.nan
    return g
