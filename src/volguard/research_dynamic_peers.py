"""Past-only monthly price peers on audited observations; no sector/causal claim."""
import numpy as np
import pandas as pd

COLUMNS=['peer_return1','peer_return5','peer_relative20','peer_volume','peer_gap1','peer_gap5']
def build(frame):
    fields=['return_1','return_5','relative20','volume_ratio']
    f=frame.sort_values(['date','symbol']).copy()
    if f.duplicated(['date','symbol']).any():raise ValueError('Duplicate peer input')
    matrices={c:f.pivot(index='date',columns='symbol',values=c) for c in fields}
    panel=matrices['return_1'];days=panel.index;symbols=panel.columns
    outputs={c:pd.DataFrame(np.nan,index=days,columns=symbols) for c in COLUMNS[:4]}
    networks=[]
    for month in sorted(days.to_period('M').unique()):
        current=days[days.to_period('M')==month];start=current.min()
        history=panel.loc[panel.index<start].tail(60)
        if len(history)<60:continue
        corr=history.corr(min_periods=40)
        edges={}
        for symbol in symbols:
            candidates=corr[symbol].drop(index=symbol).dropna()
            candidates=candidates[candidates>0].sort_values(ascending=False,kind='stable')
            if len(candidates)<5:continue
            neighbors=candidates.head(5).index.tolist();edges[symbol]=neighbors
            for field,dest in zip(fields,COLUMNS[:4]):
                values=matrices[field].loc[current,neighbors]
                outputs[dest].loc[current,symbol]=values.mean(axis=1).where(values.notna().sum(axis=1)>=4)
        networks.append(dict(month=str(month),first_forecast=str(start.date()),last_fit_date=str(history.index.max().date()),edges=edges))
    result=f[['date','symbol']].copy();index=pd.MultiIndex.from_frame(result[['date','symbol']])
    for col,values in outputs.items():result[col]=values.stack(future_stack=True).reindex(index).to_numpy()
    result['peer_gap1']=result.peer_return1.to_numpy()-f.return_1.to_numpy()
    result['peer_gap5']=result.peer_return5.to_numpy()-f.return_5.to_numpy()
    result['usable']=np.isfinite(result[COLUMNS]).all(axis=1)
    return result,networks
