# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:24:15 2013

@author: ed203246
"""

import os
#import pandas as pd
import numpy as np

WD = "/home/ed203246/data/2013_cati-freesurfer-roi-thickness"

## ===========================================================================
## AD vs CTL
## ===========================================================================
datasets_filepath = os.path.join(WD,"data/AD_CTL.npz")

Xy = np.load(datasets_filepath)
X = Xy["X"]
y = Xy["y"]

from sklearn.lda import LDA
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing

from epac import range_log2
from epac import Perms, CV, CVBestSearchRefit, Pipe, Methods
k_values = range_log2(X.shape[1], add_n=True)
C_values = [.1, 1, 10, 100]
n_folds = 10

LinearSVC(class_weight='auto')

# Pipelines(Anova filter + svm) for many values of k and C
#anova_svm = Methods(*[Pipe(SelectKBest(k=k),preprocessing.StandardScaler(),
#                      Methods(*[SVC(kernel="linear", C=C) for C in C_values]))
#                  for k in k_values])

anova_svm = Methods(*[Pipe(SelectKBest(k=k), preprocessing.StandardScaler(),
                      Methods(*[LinearSVC(C=C, penalty=penalty, class_weight='auto', dual=False) 
                                for C in C_values for penalty in  ['l1', 'l2']]))
                  for k in k_values])

# Take a look
print [l for l in anova_svm.walk_leaves()]

## k and C selection based on CV
anova_svm_cv = CVBestSearchRefit(anova_svm)

anova_svm_all = Methods(anova_svm, anova_svm_cv)
               
# Cross validate
cv = CV(anova_svm_all, n_folds=n_folds)
cv.fit_predict(X=X, y=y)
cv.reduce()

# {'key': SelectKBest(k=4)/StandardScaler/LinearSVC(penalty=l1,C=10), 'mean_score_te': 0.84, 'mean_score_tr': 0.846666666667},

pipe = Pipe(SelectKBest(k=4), preprocessing.StandardScaler(), LinearSVC(C=10, penalty="l1", 
     class_weight='auto', dual=False))
pipe.fit_predict(X=X, y=y)

sv = pipe.children[0].children[0]     
sv.estimator.coef_
kb = pipe.estimator
kb.get_support()
np.where(kb.get_support())
#Out[114]: (array([ 22,  36,  96, 146]),)

d_num.columns[np.array([ 22,  36,  96, 146])]
Index([u'LH_G_OCTEMP_MEDPARAHIP_THICKNESS', u'LH_G_TEMPORAL_INF_THICKNESS', u'RH_G_OCTEMP_MEDPARAHIP_THICKNESS', u'RH_S_TEMPORAL_SUP_THICKNESS'], dtype=object)

