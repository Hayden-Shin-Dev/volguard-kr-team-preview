"""Isolated KRX direction features with explicit calendar and continuity checks."""
import numpy as np
import pandas as pd
from .direction import COLS

def build_features(bars, calendar):
    days=pd.DatetimeIndex(calendar);blocks=[]
    for symbol,raw in bars.groupby('symbol',sort=True):
        raw=raw.sort_values('date');active=days[(days>=raw.date.min())&(days<=raw.date.max())]
        g=raw.set_index('date').reindex(active);present=g.close.notna()
        valid=present&g[['open','high','low','close']].gt(0).all(axis=1)&g.volume.gt(0)&g.high.ge(g[['open','close','low']].max(axis=1))&g.low.le(g[['open','close','high']].min(axis=1))
        previous=g.close.shift();reference=g.close-g.change
        continuous=valid&valid.shift(fill_value=False)&reference.sub(previous).abs().le(.5)
        ret=(g.close/previous-1).where(continuous)
        x=pd.DataFrame(index=active);x['symbol']=symbol;x['present']=present;x['price_valid']=valid
        x['continuity_verified']=continuous;x['reference_discontinuity']=valid&valid.shift(fill_value=False)&reference.sub(previous).abs().gt(.5)
        x['return_1']=ret
        for n in [5,10,20]:x[f'return_{n}']=np.expm1(np.log1p(ret).rolling(n).sum())
        x['vol_22']=ret.rolling(22).std();x['vol_change']=ret.rolling(5).std()/x.vol_22.replace(0,np.nan)
        log=np.log1p(ret);chain=np.exp(log.groupby(ret.isna().cumsum()).cumsum())
        x['ma_deviation']=chain/chain.rolling(20).mean()-1;x.loc[ret.isna().rolling(20).sum().gt(0),'ma_deviation']=np.nan
        span=(100*(g.high/g.low-1)).where(valid)
        for n in [1,5,22]:x[f'range_{n}']=span.rolling(n).mean()
        x['gap']=(g.open/previous-1).where(continuous);x['body_pct']=(100*(g.close/g.open-1)).where(valid)
        x['close_location']=((g.close-g.low)/(g.high-g.low).replace(0,np.nan)).where(valid);x.loc[valid&g.high.eq(g.low),'close_location']=.5
        x['volume_ratio']=g.volume.where(valid)/g.volume.where(valid).rolling(20).mean().replace(0,np.nan)
        x['turnover_ratio']=g.turnover.where(valid)/g.turnover.where(valid).rolling(20).mean().replace(0,np.nan)
        nxt=pd.Series(days,index=days).shift(-1);x['target_date']=pd.Series(active,index=active).map(nxt)
        x['target_return']=ret.shift(-1);x['target_observed']=present.shift(-1,fill_value=False)
        x['target_continuity']=continuous.shift(-1,fill_value=False);x['flat_target']=x.target_return.eq(0)
        x['label']=pd.Series(np.where(x.target_return.notna(),x.target_return.gt(0),np.nan),index=active)
        x['label_valid']=valid&x.target_continuity&x.target_return.notna()&~x.flat_target
        x.index.name='date';blocks.append(x.reset_index())
    f=pd.concat(blocks,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    market=f.groupby('date').agg(mret=('return_1','mean'),breadth=('return_1',lambda s:s.dropna().gt(0).mean()),market_range=('range_1','mean'))
    market['market_trend']=np.expm1(np.log1p(market.mret).rolling(20).sum());market['market_vol']=market.mret.rolling(20).std();market['breadth5']=market.breadth.rolling(5).mean()
    f=f.merge(market.drop(columns='mret'),on='date',validate='many_to_one');f['relative20']=f.return_20-f.market_trend
    f['feature_valid']=f[COLS].notna().all(axis=1);f['eligible']=f.feature_valid&f.label_valid
    return f.sort_values(['date','symbol']).reset_index(drop=True)
