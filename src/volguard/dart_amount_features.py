"""Conservative receipt-window features; uncertain events never become zero amounts."""
import numpy as np
import pandas as pd
COUNT=[f'{k}_count_{n}' for k in ['acquisition','disposal'] for n in [1,5]]
AMOUNT=[f'{k}_amount_{n}' for k in ['acquisition','disposal'] for n in [1,5]]

def event_grid(receipts,amounts,symbol,start='2014-01-01',end='2026-10-16'):
    dates=pd.date_range(pd.Timestamp(start)-pd.Timedelta(days=6),end);g=pd.DataFrame(index=dates);g.index.name='date';bad=pd.Series(0.,index=dates)
    events=receipts[receipts.report_nm.str.contains('자기주식취득결정|자기주식처분결정',regex=True)].copy();events['received']=pd.to_datetime(events.rcept_dt,format='%Y%m%d')
    a=amounts[amounts.symbol.eq(symbol)].copy();a['received']=pd.to_datetime(a.received_date,format='%Y%m%d');duplicate=a.groupby(['received','kind']).receipt.transform('size').gt(1)
    a['safe']=a.status.eq('parsed')&~duplicate
    safe_receipts=set(a.loc[a.safe,'receipt']);unsafe=events[~events.rcept_no.isin(safe_receipts)]
    bad=unsafe.received.value_counts().reindex(dates,fill_value=0).sort_index()
    for kind in ['acquisition','disposal']:
        valid=a[a.safe&a.kind.eq(kind)]
        counts=valid.groupby('received').size().reindex(dates,fill_value=0);values=valid.groupby('received').common_amount_won.sum().reindex(dates,fill_value=0)
        for n in [1,5]:
            g[f'{kind}_count_{n}']=counts.rolling(n,min_periods=1).sum().shift(1)
            g[f'{kind}_amount_{n}']=values.rolling(n,min_periods=1).sum().shift(1)
    g['event_window_safe']=bad.rolling(5,min_periods=1).sum().shift(1).eq(0)
    g['symbol']=symbol
    return g[g.index>=pd.Timestamp(start)].reset_index()
