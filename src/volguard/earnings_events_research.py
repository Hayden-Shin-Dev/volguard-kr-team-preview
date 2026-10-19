"""Research event features available only after the receipt calendar date."""
import numpy as np
import pandas as pd

NUMERIC=['revenue_qoq','revenue_yoy','margin_change_qoq','margin_change_yoy',
         'operating_margin','profit_turn_positive','profit_turn_negative']

def event_features(daily,earnings,calendar,max_age=20):
    """No same-day receipt use, no fabricated consensus and no missing-event zeros."""
    calendar=pd.DatetimeIndex(calendar).sort_values().unique()
    if daily.duplicated(['symbol','date']).any():raise ValueError('Duplicate daily keys')
    events=earnings.copy();events['receipt_day']=pd.to_datetime(events.received_date,format='%Y%m%d')
    positions=calendar.searchsorted(events.receipt_day,side='right')
    events=events.loc[positions<len(calendar)].copy();positions=positions[positions<len(calendar)]
    events['available_date']=calendar[positions];events['available_session']=positions
    assert (events.available_date>events.receipt_day).all()
    if not np.isfinite(events[NUMERIC].to_numpy(dtype=float)).all():raise ValueError('Nonfinite earnings values')
    frames=[]
    for symbol,g in daily.groupby('symbol'):
        e=events[events.symbol.eq(symbol)].sort_values(['available_date','received_date','receipt']).drop_duplicates('available_date',keep='last')
        if e.empty:continue
        x=pd.merge_asof(g.sort_values('date'),e.drop(columns='symbol'),left_on='date',right_on='available_date',direction='backward',allow_exact_matches=True)
        x['earnings_age']=calendar.get_indexer(x.date)-x.available_session
        keep=x.earnings_age.between(0,max_age-1)&x.available_date.notna()
        x=x[keep].copy()
        assert (x.receipt_day<x.date).all()
        assert (x.available_date<=x.date).all()
        frames.append(x)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
