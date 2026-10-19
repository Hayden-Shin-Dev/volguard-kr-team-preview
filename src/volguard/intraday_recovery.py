"""Five-minute path information observable before the regular close."""
import numpy as np
import pandas as pd

RECOVERY=['drawdown_depth','recovery_fraction','low_time','last_hour_recovery','positive_run_share','afternoon_return','return_asymmetry']
VOLUME=['recovery_volume_share','up_down_volume_difference','late_up_volume_share','price_volume_covariance']

def build(raw):
    g=raw.copy();g['ts']=pd.to_datetime(g.cntr_tm,format='%Y%m%d%H%M%S');g=g.sort_values('ts').drop_duplicates('ts')
    for c in ['cur_prc','trde_qty']:g[c]=pd.to_numeric(g[c],errors='coerce').abs()
    clock=g.ts.dt.strftime('%H%M');g=g[clock.between('0900','1515')];rows=[]
    for date,x in g.groupby(g.ts.dt.normalize()):
        if len(x)<55 or x.ts.iloc[0].strftime('%H%M')!='0900' or x.ts.iloc[-1].strftime('%H%M')!='1515':continue
        p=x.cur_prc.to_numpy(float);v=x.trde_qty.to_numpy(float)
        if not np.isfinite(p).all() or not np.isfinite(v).all() or (p<=0).any() or v.sum()<=0:continue
        ret=np.diff(np.log(p));dv=v[1:];depth=np.log(p/np.maximum.accumulate(p));low=int(np.argmin(p));span=np.log(p.max()/p.min());late=x.ts.dt.hour.ge(14).to_numpy();afternoon=x.ts.dt.hour.ge(13).to_numpy();up=ret>0;down=ret<0
        rows.append(dict(date=date,minute_end=x.ts.max(),drawdown_depth=float(-depth.min()),recovery_fraction=float(np.log(p[-1]/p.min())/span) if span>0 else 0.,low_time=low/(len(p)-1),last_hour_recovery=float(np.log(p[-1]/p[late].min())),positive_run_share=float(np.mean(up[1:]&up[:-1])),afternoon_return=float(np.log(p[-1]/p[afternoon][0])),return_asymmetry=float((np.maximum(ret,0)**2).sum()-(np.minimum(ret,0)**2).sum()),recovery_volume_share=float(v[low+1:].sum()/v.sum()),up_down_volume_difference=float((dv[up].sum()-dv[down].sum())/v.sum()),late_up_volume_share=float(dv[up&late[1:]].sum()/v.sum()),price_volume_covariance=float(np.mean((ret-ret.mean())*(dv/(dv.mean()+1e-12)-1)))))
    return pd.DataFrame(rows)
