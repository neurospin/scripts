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
from parsimony.estimators import LogisticRegressionL1L2TV

#from sklearn import cross_validation
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


WD = "/home/ed203246/data/2013_predict-transition-caarms"
SRC = "/home/ed203246/git/scripts/2013_predict-transition-caarms"

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)
import utils

############################################################################
## PARAMETERS
############################################################################

#MODE = "cv"
MODE = "permutations"

#DATASET = "full"
DATASET = "reduced"
 
############################################################################
## Dataset: CARMS ONLY
############################################################################
Xd, yd = utils.read_Xy(WD=WD)
Xd = Xd.drop(['PAS2gr', 'CB_EXPO'], 1)
    
if DATASET == "full":
    ALPHA, L1_RATIO = 1 / np.log2(Xd.shape[1]+1), 0.8

if DATASET == "reduced":
    keep = ["@4.3", "@5.4", "@7.4", "@7.6"]
    Xd = Xd[keep]
    ALPHA, L1_RATIO = 0.01, 0.0

print ALPHA, L1_RATIO
# Add intercept
inter= pd.DataFrame([1.]*Xd.shape[0], columns=["inter"])
Xd = pd.concat((inter, Xd), axis=1)

"""
Xd2 = Xd.copy()
Xd2["TRANSITION"] = yd
Xd2.to_csv(WD+"/data/transitionPREP_ARonly_CAARMSonly.csv", index=False)
"""

X = np.asarray(Xd)
y = np.asarray(yd)
y = y.astype(float)[:, np.newaxis]
A, n_compacts = A_empty(X.shape[1]-1)

print "X.shape:", X.shape

############################################################################
## FIXED PARAMETERS
############################################################################

#N_FOLDS = 10
N_PERMS = 10001
ALPHAS = [.01, 0.05, .1, 1/ np.log2(X.shape[1]), .5, 1, 10, 100]
    #L1_RATIOS = np.arange(0, 1.1, 0.1)
L1_RATIOS = np.arange(0, 1.1, 0.1)

############################################################################
## Model selection on 10 CV
############################################################################

if MODE == "model_selection":
    RES = {alpha:{l1_ratio:dict(y_pred=[], y_true=[]) for l1_ratio in L1_RATIOS} for alpha in ALPHAS}                
    for fold, (train, test) in enumerate(utils.CV10):
        print "fold",fold
        Xtr = X[train, :]
        Xte = X[test, :]
        ytr = y[train, :]
        yte = y[test, :]
        for alpha in ALPHAS:
            for l1_ratio in L1_RATIOS:
                k, l, g = alpha * np.array([1-l1_ratio, l1_ratio, 0])
                mod = LogisticRegressionL1L2TV(k=k, l=l, g=g, A=A,
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
    res.to_csv(os.path.join(WD, "enet_param-selection_10cv.csv"), index=False)

# =>
#"alpha"      "l1_ratio" "speci", "senci",                "bcr"
#0.2127460536    	0.8	0.9375	  0.8181818182	0.8778409091

############################################################################
## 10 CV with parameters scores + AUC and ROC
############################################################################
if MODE == "cv":
    #ALPHA, L1_RATIO
    #alpha, l1_ratio = 1 / np.log2(X.shape[1]), 0.8
    y_pred = list()
    y_true = list()
    y_prob_pred = list()
    for fold, (train, test) in enumerate(utils.CV10):
        Xtr = X[train, :]
        Xte = X[test, :]
        ytr = y[train, :]
        yte = y[test, :]
        print "fold",fold
        k, l, g = ALPHA * np.array([1-L1_RATIO, L1_RATIO, 0])
        mod = LogisticRegressionL1L2TV(k=k, l=l, g=g, A=A,
                                            penalty_start=1,
                                            class_weight="auto")
        mod.fit(Xtr, ytr)
        y_pred.append(mod.predict(Xte).ravel())
        y_prob_pred.append(mod.predict_probability(Xte).ravel())
        y_true.append(yte.ravel())
        #print mod.predict(Xte).ravel() == yte.ravel()
        #if fold == 4: break
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y_prob_pred = np.concatenate(y_prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pred)
    roc_auc = auc(fpr, tpr)
    print "Precisions: ", p, p.mean(), "\n",\
          "Recalls:    ", r, r.mean(), "\n",\
          "F:          ", f, f.mean(), "\n",\
          "AUC:        ", roc_auc, "\n",\
          "Support:    ", s
    import pyroc
    sample = np.c_[y_true, y_prob_pred]
    roc = pyroc.ROCData(sample)  #Create the ROC Object
    roc.auc() #get the area under the curve
    # 0.9829545454545454
    roc.plot('ROC Curve (AUC= %.2f)' % roc.auc(), True, True) #Create a plot of the ROC curve



############################################################################
## Permuations + 10 CV
############################################################################
if MODE == "permutations":
    k, l, g = ALPHA * np.array([1 - L1_RATIO, L1_RATIO, 0])
    mod = LogisticRegressionL1L2TV(k=k, l=l, g=g, A=A,
                                                    penalty_start=1,
                                                    class_weight="auto")
    
    scores = list()
    coefs_count = list()
    coefs_mean = list()
    coefs_std = list()
    for perm_i in xrange(N_PERMS):
        print "** Perm", perm_i
        if perm_i == 0:
            yp = y
        else:
            yp = y[np.random.permutation(y.shape[0])]
        #cv = cross_validation.StratifiedKFold(y=yp.ravel(), n_folds=N_FOLDS)
        cv = utils.CV10
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
        coefs_mean.append(np.mean(Coefs, axis=0))
        coefs_std.append(np.std(Coefs, axis=0))

    scores = np.r_[scores]
    coefs_count = np.r_[coefs_count]
    coefs_mean = np.r_[coefs_mean]
    coefs_std = np.r_[coefs_std]
    # avoid division by 0
    coefs_std[coefs_mean == 0] = 1.
    coefs_count_pval = np.sum(coefs_count[1:, :] >= coefs_count[0, :],  axis=0) / float(coefs_count.shape[0] - 1)
    coefs_mean_pval = np.sum(np.abs(coefs_mean[1:, :]) >= np.abs(coefs_mean[0, :]),  axis=0) / float(coefs_mean.shape[0] - 1)
    coefs_z_pval = np.sum(
        np.abs(coefs_mean[1:, :] / coefs_std[1:, :]) >= \
        np.abs(coefs_mean[0, :] / coefs_std[0, :]),  axis=0) \
        / float(coefs_mean.shape[0] - 1)
    scores_pval = np.sum(scores[1:, :] >= scores[0, :],  axis=0) / float(scores.shape[0] - 1)
    #
    nzero = coefs_count[0, ] !=0
    coefs = pd.DataFrame(dict(var=Xd.columns[nzero],
            count=coefs_count[0, nzero], count_pval=coefs_count_pval[nzero],   
            mean=coefs_mean[0, nzero],    mean_pval=coefs_mean_pval[nzero],
            z=coefs_mean[0, nzero] / coefs_std[0, nzero],
            z_pval=coefs_z_pval[nzero]))
    print coefs.to_string()
    coefs.to_csv(os.path.join(WD, "enet_permuation%i_10cv_%s-dataset_coefs.csv" %\
        (N_PERMS, DATASET)), index=False)

    names = ["prec_0", "prec_1", "prec_mean", "recall_0", "recall_1", "recall_mean",
             "f_0", "f_1", "f_mean", "support_0", "suppor_1", "auc"]
    
    scores = pd.DataFrame(
    [[names[i], scores[0, i], scores_pval[i]] for i in xrange(len(names))],
    columns=["score", "val", "p-val"])
    print scores.to_string()
    scores.to_csv(os.path.join(WD, "enet_permuation%i_10cv_%s-dataset_scores.csv" %\
        (N_PERMS, DATASET)), index=False)

"""
RESULTS


Full dataset
============


10CV
----
Precisions:  [ 0.88235294  0.9       ] 0.891176470588 
Recalls:     [ 0.9375      0.81818182] 0.877840909091 
F:           [ 0.90909091  0.85714286] 0.883116883117 
AUC:         0.835227272727 
Support:     [16 11]


NPERM = 10000
-------------

   coef_count    pval    var
0          10  1.0000  inter
1           4  0.2613   @1.2
2          10  0.0415   @4.3
3          10  0.0705   @5.4
4          10  0.0954   @7.4
5          10  0.0757   @7.6
6           7  0.1559   @7.7
          score        val   p-val
0        prec_0   0.882353  0.0069
1        prec_1   0.900000  0.0008
2     prec_mean   0.891176  0.0009
3      recall_0   0.937500  0.0024
4      recall_1   0.818182  0.0336
5   recall_mean   0.877841  0.0008
6           f_0   0.909091  0.0008
7           f_1   0.857143  0.0008
8        f_mean   0.883117  0.0008
9     support_0  16.000000  1.0000
10     suppor_1  11.000000  1.0000
11          auc   0.835227  0.0065


NPERM = 2000
------------


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


Reduced dataset
===============

10CV:
Precisions:  [ 0.88235294  0.9       ] 0.891176470588 
Recalls:     [ 0.9375      0.81818182] 0.877840909091 
F:           [ 0.90909091  0.85714286] 0.883116883117 
AUC:         0.9375 
Support:     [16 11]

NPERM = 10000
-------------
   coef_count  pval    var
0          10     1  inter
1          10     1   @4.3
2          10     1   @5.4
3          10     1   @7.4
4          10     1   @7.6
          score        val   p-val
0        prec_0   0.882353  0.0011
1        prec_1   0.900000  0.0000
2     prec_mean   0.891176  0.0000
3      recall_0   0.937500  0.0001
4      recall_1   0.818182  0.0138
5   recall_mean   0.877841  0.0000
6           f_0   0.909091  0.0000
7           f_1   0.857143  0.0000
8        f_mean   0.883117  0.0000
9     support_0  16.000000  1.0000
10     suppor_1  11.000000  1.0000
11          auc   0.937500  0.0001


"""
