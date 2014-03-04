# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 22:21:53 2014

@author: edouard
"""

import os.path
import sys
import numpy as np
import pylab as plt
import pandas as pd
from sklearn.svm import LinearSVC as SVM
from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support

#from sklearn.linear_model import LogisticRegression
#from sklearn import preprocessing
#from sklearn.feature_selection import SelectKBest


WD = "/home/edouard/data/2013_predict-transition-caarms"
SRC = "/home/edouard/git/scripts/2013_predict-transition-caarms"

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)
import IO

############################################################################
## Dataset: CARMS ONLY
## Algo: SVM L1, C=0.05
############################################################################

Xd, yd = IO.read_Xy(WD=WD)
Xd, yd = IO.read_Xy(WD=WD)
Xd = Xd.drop(['PAS2gr', 'CB_EXPO'], 1)


X = np.asarray(Xd)
y = np.asarray(yd)
svm = SVM(dual=False, class_weight='auto', penalty="l1", C=0.05)

############################################################################
## Fit on all
############################################################################
svm.fit(X, y)
coefs = svm.coef_.copy().ravel()

print pd.DataFrame(dict(var=Xd.columns[coefs !=0], coef=coefs[coefs!=0]))
#       coef   var
#0 -0.084407  @4.3
#1  0.138805  @5.4
#2 -0.113876  @7.4
#3  0.063419  @7.6
#4  0.011631  @7.7
#

############################################################################
## 10 CV
############################################################################

cv = cross_validation.StratifiedKFold(y=y, n_folds=10)
y_pred = list()
y_true = list()
Coefs = list()
for train, test in cv:
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    svm.fit(Xtr, ytr)
    y_pred.append(svm.predict(Xte))
    y_true.append(yte)
    Coefs.append(svm.coef_.copy())

y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
Coefs = np.concatenate(Coefs)

coefs_count = np.sum(Coefs !=0, axis=0)

print pd.DataFrame(dict(var=Xd.columns[coefs_count !=0], coef_count=coefs_count[coefs_count!=0]))
#   coef_count   var
#0           1  @1.1
#1          10  @4.3
#2          10  @5.4
#3          10  @7.4
#4          10  @7.6
#5           5  @7.7

print precision_recall_fscore_support(y_true, y_pred, average=None)
#array([ 0.88235294,  0.9       ]),
#array([ 0.9375    ,  0.81818182]),
#array([ 0.90909091,  0.85714286]),
#array([16, 11]))
p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

############################################################################
## Bootstrap model
############################################################################

n_boot = 10000
np.random.seed(1)


Coefs = list()
for boot in xrange(n_boot):
    train = np.random.randint(X.shape[0], size=X.shape[0])
    Xtr = X[train, :]
    ytr = y[train, :]
    svm.fit(Xtr, ytr)
    Coefs.append(svm.coef_.copy())


Coefs = np.concatenate(Coefs)
coefs_count = np.sum(Coefs !=0, axis=0)
coefs_sd = np.std(Coefs, axis=0)
coefs_mean = np.mean(Coefs, axis=0)

mask = coefs_count>n_boot/2
print pd.DataFrame(dict(var=Xd.columns[mask], count=coefs_count[mask], mean=coefs_mean[mask], sd=coefs_sd[mask]))
#   count      mean        sd   var
#0   7919 -0.073536  0.056182  @4.3
#1   7379  0.088492  0.076817  @5.4
#2   8067 -0.089201  0.071014  @7.4
#3   6612  0.062118  0.067923  @7.6


#np.where(mask)
#(array([10, 14, 20, 22]),)

plt.hist(Coefs[:, 10], bins=50)
plt.hist(Coefs[:, 14])
plt.hist(Coefs[:, 20])
plt.hist(Coefs[:, 22])


