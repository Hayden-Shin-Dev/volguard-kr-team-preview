"""Immutable research issuance from audited, externally prepared daily features.

This module neither fetches data nor changes the selected production model.
"""
from datetime import datetime
import numpy as np
import pandas as pd
from .config import KST
from .daily_sequence_candidate import make_windows
from .direction_contract import window,aware,validate_inputs,record
from .util import digest


def issue_prepared(root,model,features,calendar,anchor,symbols,closes,evidence,*,clock=None):
    now=aware(clock or datetime.now(KST));dates=window(calendar,anchor)
    if now.date().isoformat()!=anchor or now<aware(dates['feature_cutoff']):
        raise ValueError('Same-day after-close issuance only; historical replay is not issuance')
    validate_inputs(evidence,dates['feature_cutoff'],now)
    # No target, label or future-universe eligibility is used to form input windows.
    names,x,rejected=make_windows(features,anchor,[s['date'] for s in calendar['sessions']])
    wanted=list(dict.fromkeys(symbols));positions={symbol:i for i,symbol in enumerate(names)}
    accepted=[s for s in wanted if s in positions]
    for symbol in accepted:
        close=float(closes.get(symbol,float('nan')))
        if not np.isfinite(close) or close<=0:raise ValueError('Verified anchor close missing')
    probabilities=model.predict(x[[positions[s] for s in accepted]])
    if len(probabilities)!=len(accepted) or not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
        raise ValueError('Invalid candidate probabilities')
    model_version='daily-sequence-research-'+model.version
    results=[]
    for symbol,probability in zip(accepted,probabilities):
        values=x[positions[symbol]]
        details=dict(model='daily_sequence_three_seed',model_hash=model.version,
                     anchor_close=float(closes[symbol]),input_hash=digest(values.tolist()),
                     research_only=True,automatic_promotion=False)
        try:
            path=record(root,calendar,anchor,symbol,model_version,float(probability),evidence,now=now,details=details)
            results.append(dict(symbol=symbol,status='issued',path=str(path)))
        except FileExistsError:results.append(dict(symbol=symbol,status='already_issued'))
    for symbol in wanted:
        if symbol not in positions:results.append(dict(symbol=symbol,status='unavailable',reason=rejected.get(symbol,'No audited feature history')))
    return dict(anchor_date=anchor,target_date=dates['target_date'],research_only=True,results=results)
