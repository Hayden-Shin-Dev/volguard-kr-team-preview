"""Receipt-specific event metadata, strictly previous calendar date only."""
import pandas as pd
BROAD={'capital':'증자|전환사채|자기주식','earnings':'잠정|실적','correction':'정정','contract':'단일판매|공급계약'}
SPLIT={'buyback':'자기주식취득','disposal':'자기주식처분','retirement':'주식소각','equity_issue':'유상증자결정','convertible':'전환사채권발행|신주인수권부사채권발행','earnings':'잠정|실적','correction':'정정','contract':'단일판매|공급계약'}

def columns(group):return [group+'_'+k+'_'+str(n) for k in (BROAD if group=='broad' else SPLIT) for n in [1,5]]

def build(events,symbol,start='2010-01-01',end='2026-10-16'):
    e=events.drop_duplicates('rcept_no').copy();e['received']=pd.to_datetime(e.rcept_dt,format='%Y%m%d');dates=pd.date_range(pd.Timestamp(start)-pd.Timedelta(days=6),end)
    result=pd.DataFrame({'date':dates})
    for group,patterns in [('broad',BROAD),('split',SPLIT)]:
        for name,pattern in patterns.items():
            count=e.loc[e.report_nm.str.contains(pattern,na=False),'received'].value_counts().reindex(dates,fill_value=0).sort_index()
            for n in [1,5]:result[group+'_'+name+'_'+str(n)]=count.rolling(n,min_periods=1).sum().shift(1).to_numpy()
    result['symbol']=symbol
    return result[result.date.ge(start)].reset_index(drop=True)
