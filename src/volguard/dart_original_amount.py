"""Strict amount extraction; ambiguous tables and unsupported units are rejected."""
import re

def extract_amount(text,kind):
 if kind not in ['acquisition','disposal']:raise ValueError('Unknown kind')
 verb='취득' if kind=='acquisition' else '처분'
 compact=re.sub(r'[^\S\r\n]+','',text.replace('&cr;',''))
 marker=verb+'예정금액(원)'
 if compact.count(marker)!=1:return {'status':'ambiguous_or_missing_won_table'}
 section=compact.split(marker,1)[1]
 match=re.match(r'\s*보통주(?:식)?\s*([0-9,]+)\s*기타주(?:식)?\s*([0-9,]+|-)',section)
 if not match:return {'status':'unsupported_table'}
 common=int(match.group(1).replace(',',''));other=0 if match.group(2)=='-' else int(match.group(2).replace(',',''))
 if common<=0 or common>10**16:return {'status':'invalid_common_amount'}
 return {'status':'parsed','common_amount_won':common,'other_amount_won':other,'amount_evidence':marker+match.group(0),'unit':'KRW'}
