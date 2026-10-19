"""Descriptive adjacent-band ordering with shared date-block resamples."""
import numpy as np
import pandas as pd

BINS=[.5,.55,.6,.65,.7,.8,.9,1.]


def adjacent_bands(frame,p,side,dates,indices):
    p=np.asarray(p,dtype=float)
    if len(p)!=len(frame) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
    if side not in ['up','down'] or not frame.label.isin([0,1]).all():raise ValueError('Invalid outcome or side')
    q=p if side=='up' else 1-p;y=frame.label.to_numpy(float) if side=='up' else 1-frame.label.to_numpy(float)
    groups=[]
    for low,high in zip(BINS[:-1],BINS[1:]):
        keep=(q>=low)&((q<high)|((high==1)&(q==1)))
        d=pd.DataFrame({'date':frame.target_date.to_numpy()[keep],'y':y[keep],'n':1}).groupby('date')[['y','n']].sum().reindex(dates,fill_value=0)
        groups.append(d.to_numpy(float))
    result=[]
    for k,(a,b) in enumerate(zip(groups[:-1],groups[1:])):
        common=(a[:,1]>0)&(b[:,1]>0);na=a[:,1].sum();nb=b[:,1].sum()
        row=dict(side=side,lower_band=[BINS[k],BINS[k+1]],upper_band=[BINS[k+1],BINS[k+2]],lower_n=int(na),upper_n=int(nb),lower_dates=int((a[:,1]>0).sum()),upper_dates=int((b[:,1]>0).sum()),common_dates=int(common.sum()),frequency_difference=None,frequency_difference_ci95=None,same_date_equal_weight_difference=None,same_date_difference_ci95=None)
        if na and nb:
            row['frequency_difference']=float(b[:,0].sum()/nb-a[:,0].sum()/na)
            aa=a[indices].sum(axis=1);bb=b[indices].sum(axis=1);valid=(aa[:,1]>0)&(bb[:,1]>0)
            if len(dates)>=40 and min(row['lower_dates'],row['upper_dates'])>=20 and valid.mean()>=.95:
                row['frequency_difference_ci95']=np.quantile(bb[valid,0]/bb[valid,1]-aa[valid,0]/aa[valid,1],[.025,.975]).tolist()
        if common.any():
            delta=np.zeros(len(dates));delta[common]=b[common,0]/b[common,1]-a[common,0]/a[common,1]
            row['same_date_equal_weight_difference']=float(delta[common].mean())
            count=common[indices].sum(axis=1);valid=count>0
            if len(dates)>=40 and common.sum()>=20 and valid.mean()>=.95:
                row['same_date_difference_ci95']=np.quantile(delta[indices][valid].sum(axis=1)/count[valid],[.025,.975]).tolist()
        result.append(row)
    return result
