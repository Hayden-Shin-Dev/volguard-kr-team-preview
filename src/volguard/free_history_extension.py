"""Paired three-stock history-length diagnostic, not a production candidate."""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .free_information_study import CORE, setup, number
from .direction_calibration_study import fit_calibration, calibrated
from .direction_historical_reliability import compare, draws
from .probability_quality import evaluate
from .event_direction import freeze
from .util import atomic_json, utcnow


def run():
    root, out, _ = setup()
    freeze(out / 'history-protocol.json', dict(train_starts=['2000-01-01','2010-01-01'], train_end='2021-12-31', calibration='2022 sigmoid', evaluation='2024 reused; fixed paired history-length diagnostic', features=['return1','return5','return20','vol20','range','body','gap','volume_ratio'], caveat='Kiwoom adjusted daily retrospective vintage; corporate-action history not fully certified; core three survivors only'))
    parts, audit = [], []
    krx = pd.read_parquet(root/'silver/bars.parquet')
    cols=['return1','return5','return20','vol20','range','body','gap','volume_ratio']
    for symbol in CORE:
        f=pd.read_parquet(out/f'bars-{symbol}-daily.parquet').drop_duplicates('dt').sort_values('dt')
        f['date']=pd.to_datetime(f.dt)
        for old,new in [('cur_prc','close'),('open_pric','open'),('high_pric','high'),('low_pric','low'),('trde_qty','volume')]:
            f[new]=number(f[old]).abs()
        valid=(f.low>0)&(f.low<=f[['open','close']].min(axis=1))&(f.high>=f[['open','close']].max(axis=1))&(f.volume>0)
        f.loc[~valid,['close','open','high','low','volume']]=np.nan
        f['return1']=f.close.pct_change(fill_method=None)
        for n in [5,20]: f[f'return{n}']=f.close.pct_change(n,fill_method=None)
        f['vol20']=f.return1.rolling(20).std()
        f['range']=f.high/f.low-1
        f['body']=f.close/f.open-1
        f['gap']=f.open/f.close.shift()-1
        f['volume_ratio']=f.volume/f.volume.shift().rolling(20).mean()
        f['target_date']=f.date.shift(-1)
        f['outcome']=f.close.shift(-1)/f.close-1
        f['label']=(f.outcome>0).astype(int)
        f['symbol']=symbol
        # Audit exact outcome sign against KRX adjusted-close history where available.
        k=krx[krx.symbol.eq(symbol)].sort_values('date').copy()
        k['krx_return']=k.adj_close.shift(-1)/k.adj_close-1
        k['krx_target']=k.date.shift(-1)
        merged=f.merge(k[['date','krx_return','krx_target']],on='date',how='left',validate='one_to_one')
        paired=merged.krx_return.notna()&merged.target_date.eq(merged.krx_target)&merged.outcome.notna()
        mismatch=paired & (np.sign(merged.outcome)!=np.sign(merged.krx_return))
        audit.append(dict(symbol=symbol,rows=len(f),overlap=int(paired.sum()),direction_mismatches=int(mismatch.sum())))
        f=merged.loc[~mismatch].copy()
        # Corporate-action/invalid return neighborhoods excluded conservatively.
        unsafe=f.return1.abs().gt(.32).rolling(22,min_periods=1).max().eq(1)|f.outcome.abs().gt(.32)
        f=f[~unsafe & f.outcome.ne(0)&f.outcome.notna()].dropna(subset=cols)
        parts.append(f)
    f=pd.concat(parts,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna(subset=cols)
    cal=f[f.target_date.dt.year.eq(2022)]
    test=f[f.target_date.dt.year.eq(2024)]
    reference_train=f[f.target_date.between('2010-01-01','2021-12-31')]
    prior=float(reference_train.label.mean())
    result, predictions={},test[['symbol','date','target_date','label']].copy()
    with threadpool_limits(limits=3):
        for start in ['2010-01-01','2000-01-01']:
            t=f[f.target_date.between(start,'2021-12-31')]
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=700,random_state=17))
            model.fit(t[cols],t.label)
            c=fit_calibration(cal,model.predict_proba(cal[cols])[:,1],'sigmoid')
            p=calibrated(c,test,model.predict_proba(test[cols])[:,1],'sigmoid')
            predictions[start]=p
            result[start]=dict(train_rows=len(t),**evaluate(test,p,prior))
    dates=sorted(test.target_date.unique())
    report=dict(at=utcnow(),results=result,price_audit=audit,comparison_2000_minus_2010=compare(test,predictions['2010-01-01'].to_numpy(),predictions['2000-01-01'].to_numpy(),dates,draws(dates)),independent_test=False,production_changes=False,scope='Three-stock reduced daily feature set; not comparable to full-universe headline scores')
    predictions.to_parquet(out/'history-predictions.parquet',index=False)
    atomic_json(out/'history-report.json',report)
    print(json.dumps({k:dict(train=v['train_rows'],brier=v['date_balanced']['brier'],log_loss=v['date_balanced']['log_loss'],strong=v['high60_coverage']) for k,v in result.items()}),flush=True)


if __name__=='__main__': run()
