"""Proper log-loss offset model; no probability expansion or confidence floor."""
import numpy as np
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def objective(coef,x,y,offset,weights,alpha):
 score=offset+x@coef[:-1]+coef[-1];err=(expit(score)-y)*weights
 return np.mean(weights*(np.logaddexp(0,score)-y*score))+alpha*(coef@coef)/2,np.r_[x.T@err/len(y),err.mean()]+alpha*coef

def fit(frame,columns,alpha):
 scaler=StandardScaler().fit(frame[columns]);x=scaler.transform(frame[columns]);offset=logit(np.clip(frame.baseline,1e-6,1-1e-6));w=1/frame.target_date.map(frame.target_date.value_counts()).to_numpy();w/=w.mean()
 opt=minimize(objective,np.zeros(len(columns)+1),args=(x,frame.label.to_numpy(),offset,w,alpha),jac=True,method='L-BFGS-B')
 if not opt.success:raise ValueError('Residual fit did not converge')
 return {'scaler':scaler,'coef':opt.x,'columns':columns,'alpha':alpha,'fit_last_target':frame.target_date.max(),'research_only':True}

def predict(model,frame):
 if not model['fit_last_target']<frame.date.min():raise ValueError('Prediction not after training outcomes')
 score=logit(np.clip(frame.baseline,1e-6,1-1e-6))+model['scaler'].transform(frame[model['columns']])@model['coef'][:-1]+model['coef'][-1]
 return expit(score)
