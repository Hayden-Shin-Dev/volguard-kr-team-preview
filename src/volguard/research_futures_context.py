"""Official regular-session futures context. Historical publication times unverified."""
import numpy as np
import pandas as pd


def summarize(rows):
    f = pd.DataFrame(rows)
    f = f[(f.MKT_NM == '정규') & (f.PROD_NM == '코스피200 선물')].copy()
    if f.empty:
        return None
    if f.BAS_DD.nunique() != 1 or f.ISU_CD.duplicated().any():
        raise ValueError('Duplicate contracts or mixed dates')
    cols = ['TDD_CLSPRC', 'SPOT_PRC', 'ACC_TRDVOL', 'ACC_OPNINT_QTY',
            'TDD_OPNPRC', 'CMPPREVDD_PRC']
    for c in cols:
        f[c] = pd.to_numeric(f[c].astype(str).str.replace(',', ''), errors='coerce')
    valid = np.isfinite(f[cols]).all(axis=1) & (f.ACC_TRDVOL > 0) & (f.SPOT_PRC > 0) & (f.TDD_OPNPRC > 0) & (f.TDD_CLSPRC > 0) & (f.ACC_OPNINT_QTY >= 0)
    f = f[valid]
    if f.empty:
        return None
    w = f.ACC_TRDVOL / f.ACC_TRDVOL.sum()
    return dict(date=pd.Timestamp(f.BAS_DD.iloc[0]),
                basis=float(w @ (f.TDD_CLSPRC / f.SPOT_PRC - 1)),
                intraday=float(w @ (f.TDD_CLSPRC / f.TDD_OPNPRC - 1)),
                volume=float(f.ACC_TRDVOL.sum()), open_interest=float(f.ACC_OPNINT_QTY.sum()),
                active_contracts=len(f))


def align_previous_session(summary, calendar):
    """Exact previous observed trading session only; missing dates never forward-filled."""
    dates = pd.DatetimeIndex(calendar).sort_values().unique()
    if summary.date.duplicated().any():
        raise ValueError('Duplicate source dates')
    f = summary.set_index('date').reindex(dates)
    f['volume_change'] = np.log(f.volume.where(f.volume > 0)).diff()
    f['oi_change'] = np.log(f.open_interest.where(f.open_interest > 0)).diff()
    f['source_date'] = f.index
    f = f.shift(1)
    f.index.name = 'date'
    return f.reset_index()


def relative_context(aligned):
    """Normalize known prior-session observations using strictly older sessions."""
    f = aligned.sort_values('date').copy()
    scale = f.intraday.rolling(60, min_periods=40).std().shift(1)
    normal_volume = f.volume.rolling(60, min_periods=40).median().shift(1)
    f['intraday'] = f.intraday / scale.where(scale > 0)
    f['volume_change'] = np.log(f.volume.where(f.volume > 0) / normal_volume.where(normal_volume > 0))
    f['usable'] = np.isfinite(f[['basis', 'intraday', 'volume_change', 'oi_change']]).all(axis=1)
    return f
