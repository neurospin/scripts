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
N_PERMS = 10000
N_FOLDS = 10
N_BOOTS = 10000
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
## Permuations + 10 CV
############################################################################

recalls = list()
precisions = list()
coefs_count = list()
for perm_i in xrange(N_PERMS):
    if perm_i == 0:
        yp = y
    else:
        yp = y[np.random.permutation(y.shape[0])]
    cv = cross_validation.StratifiedKFold(y=yp, n_folds=N_FOLDS)
    y_pred = list()
    y_true = list()
    Coefs = list()
    for train, test in cv:
        Xtr = X[train, :]
        Xte = X[test, :]
        ytr = yp[train, :]
        yte = yp[test, :]
        svm.fit(Xtr, ytr)
        y_pred.append(svm.predict(Xte))
        y_true.append(yte)
        Coefs.append(svm.coef_.copy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    Coefs = np.concatenate(Coefs)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)    
    coefs_count.append(np.sum(Coefs !=0, axis=0))
    recalls.append(np.r_[r, [r.mean()]])
    precisions.append(np.r_[p, [p.mean()]])
    
recalls = np.r_[recalls]
precisions = np.r_[precisions]
coefs_count = np.r_[coefs_count]

coefs_count_pval = np.sum(coefs_count[1:, :] > coefs_count[0, :],  axis=0) / float(coefs_count.shape[0] - 1)

recalls_pval = np.sum(recalls[1:, :] > recalls[0, :],  axis=0) / float(recalls.shape[0] - 1)

precisions_pval = np.sum(precisions[1:, :] > precisions[0, :],  axis=0) / float(precisions.shape[0] - 1)
#
nzero = coefs_count[0, ] !=0
print pd.DataFrame(dict(var=Xd.columns[nzero], coef_count=coefs_count[0, nzero], pval=coefs_count_pval[nzero]))
print pd.DataFrame(dict(recall=recalls[0, :], pval=recalls_pval))
print pd.DataFrame(dict(precision=precisions[0, :], pval=precisions_pval))
"""
N_PERMS = 10000

   coef_count      pval   var
0           1  0.065607  @1.1
1          10  0.000000  @4.3
2          10  0.000000  @5.4
3          10  0.000000  @7.4
4          10  0.000000  @7.6
5           5  0.152215  @7.7
       pval    recall
0  0.010701  0.937500
1  0.019702  0.818182
2  0.000200  0.877841
   precision      pval
0   0.882353  0.010801
1   0.900000  0.000700
2   0.891176  0.000300

support array([16, 11])
"""

############################################################################
## Bootstrap model
############################################################################
# Keep only 
keep = ["@4.3", "@5.4", "@7.4", "@7.6"]
Xdr = Xd[keep]

X = np.asarray(Xdr)
y = np.asarray(yd)
svm.fit(X, y)
coefs = svm.coef_.copy().ravel()
print pd.DataFrame(dict(var=Xdr.columns, coef=coefs))
"""
       coef   var
0 -0.081227  @4.3
1  0.140782  @5.4
2 -0.112707  @7.4
3  0.065021  @7.6
"""

np.random.seed(1)


Coefs = list()
for boot in xrange(N_BOOTS):
    train = np.random.randint(X.shape[0], size=X.shape[0])
    Xtr = X[train, :]
    ytr = y[train, :]
    svm.fit(Xtr, ytr)
    Coefs.append(svm.coef_.copy())


Coefs = np.concatenate(Coefs)
coefs_count = np.sum(Coefs !=0, axis=0)
coefs_sd = np.std(Coefs, axis=0)
coefs_mean = np.mean(Coefs, axis=0)

mask = coefs_count>N_BOOTS/2
print pd.DataFrame(dict(var=Xd.columns[mask], count=coefs_count[mask], mean=coefs_mean[mask], sd=coefs_sd[mask]))
"""
Boostrap with all 27 variables

   count      mean        sd   var
0   7919 -0.073536  0.056182  @4.3
1   7379  0.088492  0.076817  @5.4
2   8067 -0.089201  0.071014  @7.4
3   6612  0.062118  0.067923  @7.6

Boostrap on reduced data: with only 4 variables
   count      mean        sd   var
0   8977 -0.076910  0.047413  @1.1
1   8476  0.112727  0.075981  @1.2
2   8304 -0.095771  0.072051  @1.3
3   7480  0.076048  0.071188  @2.1
"""

#np.where(mask)
#(array([0, 1, 2, 3]),)

plt.hist(Coefs[:, 0], bins=50)
plt.hist(Coefs[:, 1], bins=50)
plt.hist(Coefs[:, 2], bins=50)
plt.hist(Coefs[:, 3], bins=50)


