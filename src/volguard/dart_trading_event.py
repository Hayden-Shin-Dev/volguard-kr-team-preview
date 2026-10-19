"""Receipt-vintage event exposure on exchange sessions, not calendar-day age."""
import numpy as np
import pandas as pd

FEATURES=[f'{k}_{v}' for k in ['acquisition','disposal'] for v in ['first','later','amount_first','amount_later']]

def trading_grid(receipts,amounts,symbol,sessions):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(sessions).unique()))
    g=pd.DataFrame(0.,index=dates,columns=FEATURES);g.index.name='date'
    bad=np.zeros(len(g),dtype=bool)
    a=amounts[amounts.symbol.eq(symbol)].copy()
    duplicate=a.groupby(['received_date','kind']).receipt.transform('size').gt(1)
    good=a[a.status.eq('parsed')&~duplicate].set_index('receipt')
    for event in receipts.to_dict('records'):
        at=int(dates.searchsorted(pd.Timestamp(event['rcept_dt']),side='right'))
        stop=min(at+5,len(g))
        if event['rcept_no'] not in good.index:
            bad[at:stop]=True;continue
        row=good.loc[event['rcept_no']];kind=row.kind;amount=float(row.common_amount_won)
        if at==len(g):continue
        g.iloc[at,g.columns.get_loc(kind+'_first')]+=1
        g.iloc[at,g.columns.get_loc(kind+'_amount_first')]+=amount
        for ix in range(at+1,stop):
            g.iloc[ix,g.columns.get_loc(kind+'_later')]+=1
            g.iloc[ix,g.columns.get_loc(kind+'_amount_later')]+=amount
    g['event_safe']=~bad;g['symbol']=symbol
    return g.reset_index()

def context_features(frame):
    """Interactions are observed on forecast close; never use future outcomes."""
    f=frame.copy();names=[]
    for kind in ['acquisition','disposal']:
        for age in ['first','later']:
            c=kind+'_'+age
            for observable in ['return_1','relative20','volume_ratio','market_trend']:
                name=c+'__'+observable;f[name]=f[c]*f[observable];names.append(name)
    return f,names
