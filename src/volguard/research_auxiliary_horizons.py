"""Training-only, audited multi-session labels. No future value becomes an input."""
import numpy as np
import pandas as pd


def labels(frame,cutoff='2019-01-01'):
    result=np.zeros((len(frame),2),dtype='float32')
    masks=np.zeros((len(frame),2),dtype=bool)
    for _,g in frame.groupby('symbol',sort=False):
        cumulative=pd.Series(1.,index=g.index)
        valid=pd.Series(True,index=g.index)
        for step in range(5):
            ret=g.target_return.shift(-step)
            end=g.target_date.shift(-step)
            valid &= g.target_observed.shift(-step).eq(True)&g.target_continuity.shift(-step).eq(True)&np.isfinite(ret)&ret.gt(-1)
            valid &= g.session.shift(-step).eq(g.session+step)
            cumulative *= 1+ret
            if step+1 in [3,5]:
                col=0 if step==2 else 1
                good=valid&end.lt(pd.Timestamp(cutoff))&cumulative.ne(1)&np.isfinite(cumulative)
                result[g.index,col]=(cumulative.gt(1)&good).to_numpy(dtype='float32')
                masks[g.index,col]=good.to_numpy()
    return result,masks
