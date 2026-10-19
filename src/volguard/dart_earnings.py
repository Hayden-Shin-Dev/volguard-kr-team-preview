"""Conservative current-quarter earnings ratios from original receipt table text."""
import re
import math

def number(s):
 s=s.strip().replace(',','').replace('%','')
 if s in ['-','']:return None
 if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?',s):raise ValueError('Not a numeric table cell')
 v=float(s)
 if not math.isfinite(v):raise ValueError('Nonfinite')
 return v

def quarter_labels(lines, received_year):
 """Read explicit Q labels without inventing exact fiscal-period dates."""
 if received_year is None:raise ValueError('Receipt year required for abbreviated quarter labels')
 head=lines[:lines.index('매출액')]
 labels=[]
 for line in head:
  match=re.fullmatch(r"\(['`]?([0-9]{2}|[0-9]{4})(?:\.([1-4])[Qq]|년([1-4])분기)\)",line)
  if not match:continue
  year=int(match.group(1));quarter=int(match.group(2) or match.group(3))
  if year<100:
   year=(received_year//100)*100+year
   if year>received_year:year-=100
  labels.append((year,quarter))
 if len(labels)!=3:raise ValueError('Missing unambiguous current/previous/year-ago quarter labels')
 current,previous,year_ago=labels
 if not 0<=received_year-current[0]<=2:raise ValueError('Quarter label inconsistent with receipt year')
 def index(x):return x[0]*4+x[1]-1
 if index(current)-index(previous)!=1 or index(current)-index(year_ago)!=4:raise ValueError('Inconsistent comparable quarters')
 return dict(period_label=f'{current[0]}Q{current[1]}',previous_period_label=f'{previous[0]}Q{previous[1]}',year_ago_period_label=f'{year_ago[0]}Q{year_ago[1]}',period_basis='explicit quarter labels; exact fiscal dates not asserted')

def extract_earnings(text, *, allow_quarter_labels=False, received_year=None):
 s=text.replace('&cr;','\n');lines=[re.sub(r'\s+','',x) for x in s.splitlines() if x.strip()]
 result={};evidence={}
 try:
  for label,key in [('매출액','revenue'),('영업이익','operating_profit')]:
   starts=[i for i,x in enumerate(lines) if x==label and i+1<len(lines) and lines[i+1]=='당해실적']
   if len(starts)!=1:raise ValueError('Ambiguous table')
   i=starts[0]+2;j=lines.index('누계실적',i);cells=lines[i:j]
   if len(cells) not in [5,7]:raise ValueError('Unsupported table columns')
   # 7-column table inserts sign-transition columns after both growth rates.
   year=4 if len(cells)==7 else 3
   values=[number(cells[k]) for k in [0,1,year]]
   if any(x is None for x in values):raise ValueError('Missing comparable quarter')
   result.update({key+'_'+period:v for period,v in zip(['current','previous','year_ago'],values)})
   evidence[key]=cells
   if key=='revenue':
    if min(values)<=0:raise ValueError('Nonpositive revenue')
    for index,denom in [(2,values[1]),(year+1,values[2])]:
     reported=number(cells[index])
     if reported is not None and abs(100*(values[0]/denom-1)-reported)>.6:raise ValueError('Reported growth mismatch')
  head='\n'.join(lines)
  period=re.search(r'실적기간\n당기실적\n(\d{4}-\d{2}-\d{2})\n~\n(\d{4}-\d{2}-\d{2})',head)
  if not period:
   if not allow_quarter_labels:raise ValueError('Missing explicit quarter dates')
   result.update(quarter_labels(lines,received_year))
  else:result.update(period_start=period.group(1),period_end=period.group(2),period_basis='explicit dates')
  for horizon,key in [('qoq','previous'),('yoy','year_ago')]:
   result['revenue_'+horizon]=result['revenue_current']/result['revenue_'+key]-1
   result['margin_change_'+horizon]=result['operating_profit_current']/result['revenue_current']-result['operating_profit_'+key]/result['revenue_'+key]
  result['operating_margin']=result['operating_profit_current']/result['revenue_current']
  result['profit_turn_positive']=int(result['operating_profit_current']>0 and result['operating_profit_year_ago']<0)
  result['profit_turn_negative']=int(result['operating_profit_current']<0 and result['operating_profit_year_ago']>0)
  return {'status':'parsed','unit_policy':'Within-table ratios only; no absolute KRW conversion or consensus surprise claim','evidence':evidence,**result}
 except (ValueError,IndexError) as exc:return {'status':'unsupported','reason':str(exc),'evidence':evidence}
