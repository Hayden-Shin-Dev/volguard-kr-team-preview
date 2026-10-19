"""Research-only accounting audit for the documented2012 investor classification change."""
import pandas as pd
from volguard.free_information_study import number

CHANGE = pd.Timestamp('2012-07-16')
SOURCE = 'https://www.newswire.co.kr/newsRead.php?no=635519'
PRIVATE = ['fnnc_invt','insrnc','invtrt','etc_fnnc','bank','penfnd_etc','samo_fund']


def audit(raw):
    date=pd.to_datetime(raw.dt,format='%Y%m%d')
    n=pd.DataFrame({k:number(raw[k]) for k in ['orgn','natn','ind_invsr','frgnr_invsr','etc_corp','natfor']+PRIVATE})
    post=date.ge(CHANGE)
    broad=n[['orgn','ind_invsr','frgnr_invsr','etc_corp','natfor']].sum(axis=1,min_count=5)
    sub=n[PRIVATE].sum(axis=1,min_count=len(PRIVATE))
    result=pd.DataFrame({'date':date,'post_change':post})
    result['broad_naive_residual']=broad
    result['broad_date_aware_residual']=broad+n.natn.where(~post,0.)
    result['subtype_date_aware_residual']=sub+n.natn.where(post,0.)-n.orgn
    # This removes the reported public component; it does not claim to reverse
    # all concurrent classification changes or verify historical publication times.
    result['institution_ex_public']=n.orgn-n.natn.where(post,0.)
    result['classification_audit_scope']=date.ge('2010-01-01')
    return result
