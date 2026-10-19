"""Own-stock past-history normalization, with current observation excluded."""
import numpy as np
import pandas as pd
from .research_gross_flow import COLUMNS


def relative_features(frame, calendar):
    if frame.date.duplicated().any():raise ValueError('Duplicate dates')
    dates=pd.DatetimeIndex(calendar)
    f=frame.set_index('date').sort_index()
    dates=dates[(dates>=f.index.min())&(dates<=f.index.max())]
    values=f.reindex(dates)[COLUMNS]
    past=values.shift(1).rolling(252,min_periods=120)
    std=past.std()
    relative=(values-past.mean())/std.where(std.gt(1e-12))
    relative.columns=['relative_'+c for c in COLUMNS]
    relative['date']=relative.index
    relative['relative_usable']=np.isfinite(relative.drop(columns='date')).all(axis=1)
    return relative.reset_index(drop=True)
