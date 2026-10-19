"""Leave-one-training-year ensemble; no retrospective confidence thresholds."""
import numpy as np
from .oof_orthogonal import fit_detector as fit_one,COLUMNS

class YearEnsemble:
    def __init__(self,models,omitted_years):
        self.models=models;self.omitted_years=omitted_years
    def predict(self,x):
        return np.mean([m.predict(x) for m in self.models],axis=0)

def fit_detector(oof,kind):
    if kind=='pooled':return fit_one(oof,'centered')
    years=sorted(oof.date.dt.year.unique())
    if len(years)<2:raise ValueError('At least two training years required')
    return YearEnsemble([fit_one(oof[oof.date.dt.year.ne(y)],'centered') for y in years],years)

def combined_probability(base,detector,frame,kind,strength):
    if strength==0:return np.asarray(base)
    return np.clip(base+strength*detector.predict(frame[COLUMNS]),1e-6,1-1e-6)
