"""Version 3: match complete amount rows, not repeated narrative headings."""
import re

def extract_amount(text,kind):
 if kind not in ['acquisition','disposal']:raise ValueError('Unknown kind')
 verb='취득' if kind=='acquisition' else '처분';compact=re.sub(r'[^\S\r\n]+','',text.replace('&cr;',''));marker=verb+'예정금액(원)'
 pattern=re.escape(marker)+r'\s*보통주(?:식)?\s*([0-9,]+)\s*기타주(?:식)?\s*([0-9,]+|-)'
 matches=list(re.finditer(pattern,compact))
 if not matches:return {'status':'no_unambiguous_numeric_won_table'}
 amounts={(int(m.group(1).replace(',','')),0 if m.group(2)=='-' else int(m.group(2).replace(',',''))) for m in matches}
 if len(amounts)!=1:return {'status':'conflicting_amount_tables'}
 common,other=next(iter(amounts))
 if common<=0 or common>10**16:return {'status':'invalid_common_amount'}
 return {'status':'parsed','common_amount_won':common,'other_amount_won':other,'amount_evidence':matches[0].group(0),'unit':'KRW','matching_tables':len(matches)}
