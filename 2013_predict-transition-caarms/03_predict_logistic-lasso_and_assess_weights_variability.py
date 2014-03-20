# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 01:03:34 2014

@author: edouard
"""

import os.path
import sys
import numpy as np
import pylab as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from parsimony.estimators import RidgeLogisticRegression_L1_TV

from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support

def A_empty(p):
    """Generates empty the linear operator.
    """
    import scipy.sparse as sparse
    Ax = sparse.csr_matrix((p, p))
    Ay = sparse.csr_matrix((p, p))
    Az = sparse.csr_matrix((p, p))
    return [Ax, Ay, Az], 0

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
Xd = Xd.drop(['PAS2gr', 'CB_EXPO'], 1)
"""
Xd2 = Xd.copy()
Xd2["TRANSITION"] = yd
Xd2.to_csv(WD+"/data/transitionPREP_ARonly_CAARMSonly.csv", index=False)
"""

# Add intercept
inter= pd.DataFrame([1.]*Xd.shape[0], columns=["inter"])
Xd = pd.concat((inter, Xd), axis=1)
#Xd.shape

X = np.asarray(Xd)
y = np.asarray(yd)
y = y.astype(float)[:, np.newaxis]
A, n_compacts = A_empty(X.shape[1]-1)

############################################################################
## Model selection on 10 CV
############################################################################

cv = cross_validation.StratifiedKFold(y=yd, n_folds=10)
ALPHAS = [.01, 0.05, .1, 1/ np.log2(X.shape[1]), .5, 1, 10, 100]
#L1_RATIOS = np.arange(0, 1.1, 0.1)
L1_RATIOS = np.arange(0, 1.1, 0.1)

RES = {alpha:{l1_ratio:dict(y_pred=[], y_true=[]) for l1_ratio in L1_RATIOS} for alpha in ALPHAS}

            
for fold, (train, test) in enumerate(cv):
    print "fold",fold
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    for alpha in ALPHAS:
        for l1_ratio in L1_RATIOS:
            k, l, g = alpha * np.array([1-l1_ratio, l1_ratio, 0])
            mod = RidgeLogisticRegression_L1_TV(k=k, l=l, g=g, A=A,
                                                penalty_start=1,
                                                class_weight="auto")
            mod.fit(Xtr, ytr)
            RES[alpha][l1_ratio]["y_pred"].append(mod.predict(Xte).ravel())
            RES[alpha][l1_ratio]["y_true"].append(yte.ravel())


scores = list()
for alpha in ALPHAS:
    for l1_ratio in L1_RATIOS:
        y_pred = np.concatenate(
RES[alpha][l1_ratio]["y_pred"])
        y_true = np.concatenate(
RES[alpha][l1_ratio]["y_true"])
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        scores.append([alpha, l1_ratio] +r.tolist() + [r.mean()])


res = pd.DataFrame(scores, columns=["alpha", "l1_ratio", "speci", "senci", "bcr"])
print res
print res.to_string()
res.to_csv(os.path.join(WD, "results_enet.csv"), index=False)

# =>
#"alpha"      "l1_ratio" "speci", "senci",                "bcr"
#0.2127460536    	0.8	0.9375	  0.8181818182	0.8778409091

############################################################################
## 10 CV with parameters scores + AUC and ROC
############################################################################

cv = cross_validation.StratifiedKFold(y=yd, n_folds=10)
alpha, l1_ratio = 1 / np.log2(X.shape[1]), 0.8

y_pred = list()
y_true = list()
y_prob_pred = list()


for fold, (train, test) in enumerate(cv):
    print "fold",fold
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    k, l, g = alpha * np.array([1-l1_ratio, l1_ratio, 0])
    mod = RidgeLogisticRegression_L1_TV(k=k, l=l, g=g, A=A,
                                        penalty_start=1,
                                        class_weight="auto")
    mod.fit(Xtr, ytr)
    y_pred.append(mod.predict(Xte).ravel())
    y_prob_pred.append(mod.predict_probability(Xte).ravel())
    y_true.append(yte.ravel())


y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
y_prob_pred = np.concatenate(y_prob_pred)


p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)

fpr, tpr, thresholds = roc_curve(y_true, y_prob_pred)
roc_auc = auc(fpr, tpr)


print p, p.mean(), r, r.mean(), f, f.mean(), s, roc_auc
# =>
# [ 0.88235294  0.9       ] 0.891176470588
# [ 0.9375      0.81818182] 0.877840909091
# [ 0.90909091  0.85714286] 0.883116883117
# [16 11]
# 0.835227272727


import pyroc
sample = np.c_[y_true, y_prob_pred]
roc = pyroc.ROCData(sample)  #Create the ROC Object
roc.auc() #get the area under the curve
# 0.9829545454545454
roc.plot('ROC Curve (AUC= %.2f)' % roc.auc(), True, True) #Create a plot of the ROC curve



############################################################################
## Permuations + 10 CV
############################################################################
N_PERMS = 2001
N_FOLDS = 10

A, n_compacts = A_empty(X.shape[1] - 1)
alpha = 1 / np.log2(X.shape[1])
l1_ratio = .8
k, l, g = alpha * np.array([1 - l1_ratio, l1_ratio, 0])

mod = RidgeLogisticRegression_L1_TV(k=k, l=l, g=g, A=A,
                                                penalty_start=1,
                                                class_weight="auto")

scores = list()
coefs_count = list()
for perm_i in xrange(N_PERMS):
    print "** Perm", perm_i
    if perm_i == 0:
        yp = y
    else:
        yp = y[np.random.permutation(y.shape[0])]
    cv = cross_validation.StratifiedKFold(y=yp.ravel(), n_folds=N_FOLDS)
    y_pred = list()
    y_true = list()
    Coefs = list()
    y_prob_pred = list()
    for fold, (train, test) in enumerate(cv):
        print "fold", fold, len(test)
        Xtr = X[train, :]
        Xte = X[test, :]
        ytr = yp[train, :]
        yte = yp[test, :]
        mod.fit(Xtr, ytr)
        y_pred.append(mod.predict(Xte).ravel())
        y_true.append(yte.ravel())
        y_prob_pred.append(mod.predict_probability(Xte).ravel())
        Coefs.append(mod.beta.copy().ravel())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y_prob_pred = np.concatenate(y_prob_pred)
    Coefs = np.r_[Coefs]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pred)
    roc_auc = auc(fpr, tpr)
    scores.append(np.r_[p, p.mean(), r, r.mean(), f, f.mean(), s, roc_auc])
    coefs_count.append(np.sum(Coefs != 0, axis=0))

scores = np.r_[scores]
coefs_count = np.r_[coefs_count]

coefs_count_pval = np.sum(coefs_count[1:, :] > coefs_count[0, :],  axis=0) / float(coefs_count.shape[0] - 1)

scores_pval = np.sum(scores[1:, :] > scores[0, :],  axis=0) / float(scores.shape[0] - 1)

#
nzero = coefs_count[0, ] !=0
print pd.DataFrame(dict(var=Xd.columns[nzero], coef_count=coefs_count[0, nzero], pval=coefs_count_pval[nzero]))

names = ["prec_0", "prec_1", "prec_mean", "recall_0", "recall_1", "recall_mean",
         "f_0", "f_1", "f_mean", "support_0", "suppor_1", "auc"]

res = pd.DataFrame(
[[names[i], scores[0, i], scores_pval[i]] for i in xrange(len(names))],
columns=["score", "val", "p-val"])

print res

"""
NPERM = 2000

   coef_count    pval    var
0          10  0.0000  inter
1           4  0.2195   @1.2
2          10  0.0000   @4.3
3          10  0.0000   @5.4
4          10  0.0000   @7.4
5          10  0.0000   @7.6
6           7  0.1360   @7.7

          score        val   p-val
0        prec_0   0.882353  0.0045
1        prec_1   0.900000  0.0000
2     prec_mean   0.891176  0.0000
3      recall_0   0.937500  0.0000
4      recall_1   0.818182  0.0045
5   recall_mean   0.877841  0.0000
6           f_0   0.909091  0.0000
7           f_1   0.857143  0.0000
8        f_mean   0.883117  0.0000
9     support_0  16.000000  0.0000
10     suppor_1  11.000000  0.0000
11          auc   0.835227  0.0055

N_PERMS = 1000

   coef_count      pval    var
0          10  0.000000  inter
1           4  0.233233   @1.2
2          10  0.000000   @4.3
3          10  0.000000   @5.4
4          10  0.000000   @7.4
5          10  0.000000   @7.6
6           7  0.131131   @7.7
       pval    recall
0  0.001001  0.937500
1  0.007007  0.818182
2  0.000000  0.877841
   precision      pval
0   0.882353  0.007007
1   0.900000  0.001001
2   0.891176  0.001001

support : array([16, 11])
"""

############################################################################
## Reduced model: 10 CV with parameters scores + AUC and ROC
############################################################################
keep = ["@4.3", "@5.4", "@7.4", "@7.6"]
Xdr = Xd[keep]
# Add intercept
inter= pd.DataFrame([1.]*Xd.shape[0], columns=["inter"])
Xdr = pd.concat((inter, Xdr), axis=1)
#Xd.shape

Xr = np.asarray(Xdr)
Ar, n_compacts = A_empty(Xr.shape[1]-1)

cv = cross_validation.StratifiedKFold(y=yd, n_folds=10)
alpha, l1_ratio = 0.01, 0.0

y_pred_r = list()
y_true_r = list()
y_prob_pred_r = list()


for fold, (train, test) in enumerate(cv):
    print "fold",fold
    Xtr = Xr[train, :]
    Xte = Xr[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    k, l, g = alpha * np.array([1-l1_ratio, l1_ratio, 0])
    mod_r = RidgeLogisticRegression_L1_TV(k=k, l=l, g=g, A=Ar,
                                        penalty_start=1,
                                        class_weight="auto")
    mod_r.fit(Xtr, ytr)
    y_pred_r.append(mod_r.predict(Xte).ravel())
    y_prob_pred_r.append(mod_r.predict_probability(Xte).ravel())
    y_true_r.append(yte.ravel())

y_pred_r = np.concatenate(y_pred_r)
y_true_r = np.concatenate(y_true_r)
y_prob_pred_r = np.concatenate(y_prob_pred_r)

p, r, f, s = precision_recall_fscore_support(y_true_r, y_pred_r, average=None)
fpr, tpr, thresholds = roc_curve(y_true_r, y_prob_pred_r)
roc_auc = auc(fpr, tpr)


print p, p.mean(), r, r.mean(), f, f.mean(), s, roc_auc
# =>
# No penalization pure logistic-regression
#[ 0.875       0.81818182] 0.846590909091
#[ 0.875       0.81818182] 0.846590909091
#[ 0.875       0.81818182] 0.846590909091
#[16 11]
#0.875

# Ridge logistic regression alpha = 0.01
#[ 0.88235294  0.9       ] 0.891176470588
#[ 0.9375      0.81818182] 0.877840909091
#[ 0.90909091  0.85714286] 0.883116883117
#[16 11]
#0.9375

import pyroc
sample = np.c_[y_true, y_prob_pred]
roc = pyroc.ROCData(sample)  #Create the ROC Object
roc.auc() #get the area under the curve
# 0.9829545454545454
roc.plot('ROC Curve (AUC= %.2f)' % roc.auc(), True, True) #Create a plot of the ROC curve

############################################################################
## Compare Reduced and Full model
############################################################################
