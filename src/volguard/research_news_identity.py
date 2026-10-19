"""Conservative current-name matching for coverage audits, not historical identity proof."""
import re
import unicodedata

SUFFIXES={'co','company','ltd','limited','inc','incorporated','corp','corporation'}

def normalize(name):
    name=unicodedata.normalize('NFKC',str(name)).lower()
    words=re.sub(r'[^a-z0-9]+',' ',name).split()
    while words and words[-1] in SUFFIXES:words.pop()
    return ' '.join(words)

def alias_index(companies):
    index={};rejected=[]
    for company in companies:
        alias=normalize(company.get('corp_eng_name',''))
        if len(alias.split())<2:
            rejected.append(dict(symbol=company['symbol'],reason='empty_or_single_token_ambiguous',alias=alias))
            continue
        index.setdefault(alias,set()).add(company['symbol'])
    collisions={alias:sorted(symbols) for alias,symbols in index.items() if len(symbols)>1}
    accepted={alias:next(iter(symbols)) for alias,symbols in index.items() if len(symbols)==1}
    return accepted,rejected,collisions

def match_organizations(organizations,index):
    return sorted({index[name] for token in str(organizations).split(';')
                   if (name:=normalize(token)) in index})
