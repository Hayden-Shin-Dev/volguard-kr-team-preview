"""Scale sparse event magnitudes without amplifying rare indicator occurrences."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class EventSupportScaler(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        a=np.asarray(X,dtype=float)
        if not np.isfinite(a).all():raise ValueError('Nonfinite event inputs')
        # RMS conditional on occurrence. No centering: no event stays exactly zero.
        # Binary rare events keep scale1 instead of being multiplied by1/sqrt(prevalence).
        count=np.count_nonzero(a,axis=0)
        self.scale_=np.sqrt(np.divide((a*a).sum(axis=0),count,out=np.ones(a.shape[1]),where=count>0))
        self.scale_[self.scale_==0]=1
        self.support_=count;self.n_features_in_=a.shape[1]
        return self

    def transform(self,X):
        a=np.asarray(X,dtype=float)
        if a.shape[1]!=self.n_features_in_ or not np.isfinite(a).all():raise ValueError('Invalid event inputs')
        return a/self.scale_
