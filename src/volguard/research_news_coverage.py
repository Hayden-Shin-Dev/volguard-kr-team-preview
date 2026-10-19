"""Research-only news observation accounting. Coverage is not publication-time proof."""
import pandas as pd

def summarize(manifest, records, symbols, start, end, as_of):
    """Summarize [start,end) batches with fixed24h research lag after batch end.

    A complete archive window permits zero observed mentions, not a claim of
    zero real-world news. Incomplete windows always return null feature counts.
    """
    times=[pd.Timestamp(v) for v in [start,end,as_of]]
    if any(t.tzinfo is None for t in times):
        raise ValueError('Explicit time zones required')
    start,end,as_of=[t.tz_convert('UTC') for t in times]
    if start>=end or start.floor('15min')!=start or end.floor('15min')!=end:
        raise ValueError('Require nonempty15-minute aligned window')
    expected=pd.date_range(start,end,freq='15min',inclusive='left')
    m=pd.DataFrame(manifest).copy()
    if m.empty:m=pd.DataFrame(columns=['batch','status'])
    if m.batch.duplicated().any():raise ValueError('Duplicate manifest batch')
    m['time']=pd.to_datetime(m.batch,format='%Y%m%d%H%M%S',utc=True)
    states=m.set_index('time').status.reindex(expected)
    available=(expected+pd.Timedelta(minutes=15,hours=24))<=as_of
    covered=states.eq('ok').to_numpy()&available
    complete=bool(covered.all())
    f=records.copy()
    if f.empty:f=pd.DataFrame(columns=['batch','url','exact_symbols'])
    f['time']=pd.to_datetime(f.batch,format='%Y%m%d%H%M%S',utc=True)
    # Select before identity/URL aggregation so future copies cannot alter old counts.
    f=f[f.time.isin(expected[covered])].sort_values('time')
    result=[]
    for symbol in symbols:
        keep=f.exact_symbols.fillna('').str.split(';').map(lambda v:symbol in v).astype(bool)
        matches=f.loc[keep]
        observed=int(matches.url.nunique())
        result.append(dict(symbol=symbol,observed_unique_urls=observed,
                           feature_count=observed if complete else None))
    return dict(expected_batches=len(expected),covered_batches=int(covered.sum()),
                unavailable_batches=int((~available).sum()),complete=complete,
                counts=result,research_only=True,publication_time_verified=False)
