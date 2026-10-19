"""Auditable lexical purpose tags from receipt-original text, not model-written facts."""
import re
import pandas as pd

RULES={
 'employee':r'임직원|임원|직원|종업원|성과급|상여|근속|스톡옵션|주식매수선택권|보상',
 'shareholder':r'주주가치|주주환원|주주이익|주가안정|주가안정화',
 'retirement':r'소각',
 'funding':r'자금|재무구조|차입|부채',
 'alliance':r'제휴|협력|파트너|지분교환|상호주식|인수',
 'liquidity':r'유동성',
}
TAGS=list(RULES)+['other']
PURPOSE=[f'{kind}_{age}_{tag}' for kind in ['acquisition','disposal'] for age in ['first','later'] for tag in TAGS]

def extract_purpose(text,kind):
 verb={'acquisition':'취득','disposal':'처분'}[kind]
 s=re.sub(r'[^\S\r\n]+','',text.replace('&cr;','\n'))
 pat=verb+r'목적\s*[:：]?\s*(.{1,1500}?)(?=\n\s*\d{1,2}\.|$)'
 matches=[re.sub(r'\s+','',m).strip(':：') for m in re.findall(pat,s,re.S)]
 values=set(x for x in matches if x)
 if len(values)!=1:return {'purpose_status':'ambiguous_or_missing','purpose_text':None,'tags':[]}
 value=next(iter(values))
 tags=[tag for tag,pattern in RULES.items() if re.search(pattern,value)]
 return {'purpose_status':'parsed','purpose_text':value,'tags':tags or ['other']}

def purpose_grid(receipts,parsed,symbol,sessions):
 dates=pd.DatetimeIndex(sorted(pd.to_datetime(sessions).unique()))
 g=pd.DataFrame(0.,index=dates,columns=PURPOSE);g.index.name='date';g['purpose_safe']=True
 a=parsed[parsed.symbol.eq(symbol)].set_index('receipt')
 for event in receipts.to_dict('records'):
  at=int(dates.searchsorted(pd.Timestamp(event['rcept_dt']),side='right'));end=min(at+5,len(g))
  if event['rcept_no'] not in a.index or a.loc[event['rcept_no'],'purpose_status']!='parsed':
   g.iloc[at:end,g.columns.get_loc('purpose_safe')]=False;continue
  row=a.loc[event['rcept_no']]
  for ix in range(at,end):
   age='first' if ix==at else 'later'
   for tag in row.tags:g.iloc[ix,g.columns.get_loc(f'{row.kind}_{age}_{tag}')]+=1
 return g.assign(symbol=symbol).reset_index()
