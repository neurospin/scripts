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
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import range_log2
from epac import Perms, CV, CVBestSearchRefit, Pipe, Methods
k_values = range_log2(X.shape[1], add_n=True)
C_values = [.1, 1, 10, 100]
n_folds = 10

# Pipelines(Anova filter + svm) for many values of k and C
anova_svm = Methods(*[Pipe(SelectKBest(k=k),
                      Methods(*[SVC(kernel="linear", C=C) for C in C_values]))
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


# Add LDA
# -------

anova_svm_lda = Methods(*[Pipe(SelectKBest(k=k),
                      Methods(*[LDA()]+[SVC(kernel="linear", C=C) for C in C_values]))
                  for k in k_values])
anova_svm_lda_cv = CVBestSearchRefit(anova_svm_lda)

anova_svm_lda_all = Methods(anova_svm_lda, anova_svm_lda_cv)

# Cross validate
cv = CV(anova_svm_lda_all, n_folds=n_folds)
cv.fit_predict(X=X, y=y)
cv.reduce()
