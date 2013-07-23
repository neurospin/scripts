# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:24:15 2013

@author: ed203246
"""

import os
#import pandas as pd
import numpy as np
import time
import pandas as pd

WD = "/home/ed203246/data/2013_cati-freesurfer-roi-thickness"
csv_imaging_filepath = os.path.join(WD,"data/imaging.csv")
imaging_df = pd.read_table(csv_imaging_filepath, header=0)
imaging_variables = imaging_df.columns

## ===========================================================================
## AD vs CTL
## ===========================================================================
datasets_filepath = os.path.join(WD,"data/AD_CTL.npz")

Xy = np.load(datasets_filepath)
X = Xy["X"]
y = Xy["y"]

from sklearn.svm import LinearSVC as SVM
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing


##############################################################################
## Pipeline, "Pipe": SelectKBest + StandardScaler + SVM l1 vs l2
from epac import Pipe, CV
n_folds = 10

anova_svm = Pipe(SelectKBest(k=5), 
                 preprocessing.StandardScaler(), 
                 SVM(class_weight='auto'))

cv = CV(anova_svm, n_folds=n_folds)
cv.run(X=X, y=y)
#
res_cv_anova_svm = cv.reduce()
res_cv_anova_svm["SelectKBest/StandardScaler/LinearSVC"]['y/test/score_recall']

##############################################################################
## Multimethods, "Methods": SVM l1 vs l2
from epac import Methods, CV
svms = Methods(SVM(penalty="l1", class_weight='auto', dual=False), 
               SVM(penalty="l2", class_weight='auto', dual=False))

cv = CV(svms, n_folds=n_folds)
cv.run(X=X, y=y)
res_cv_svms = cv.reduce()
#
print res_cv_svms
print res_cv_svms["LinearSVC(penalty=l1)"]['y/test/score_recall']
print res_cv_svms["LinearSVC(penalty=l2)"]['y/test/score_recall']
# !!! BIASED RESULT !!!

# Re-fit on all data to see which mode is choosen. Warning !!! this is biased
# since all data have been used. We use for information only. No score can be
# used from it. We look the weights map.
svms.run(X=X, y=y)
print svms.children[0]
svms.children[0].estimator.coef_
print svms.children[1]
svms.children[1].estimator.coef_

print "Weights given by SVMs"
d = dict(var = imaging_variables,
svm_weights_l1 = svms.children[0].estimator.coef_.ravel(),
svm_weights_l2 = svms.children[1].estimator.coef_.ravel())
print pd.DataFrame(d).to_string()

##############################################################################
# Automatic model selection: "CVBestSearchRefit"
from epac import CVBestSearchRefit, Methods, CV

svms_auto = CVBestSearchRefit(svms)
cv = CV(svms_auto, n_folds=n_folds)
cv.run(X=X, y=y)
#
res_cv_svms_auto = cv.reduce()
print res_cv_svms_auto
print res_cv_svms_auto["CVBestSearchRefit"]['y/test/score_recall']

# Re-fit on all data. Warning: biased !!!
svms_auto.run(X=X, y=y)
print svms_auto.best_params
print svms_auto.refited.estimator.coef_

##############################################################################
# Put everything together
# Pipeline, "Pipe": SelectKBest + StandardScaler + SVM l1 vs l2
from epac import range_log2
from epac import CVBestSearchRefit, Pipe, Methods, CV
k_values = range_log2(X.shape[1], add_n=True)
C_values = [.1, 1, 10, 100]
anova_svms = Methods(*[Pipe(SelectKBest(k=k), preprocessing.StandardScaler(),
                      Methods(*[SVM(C=C, penalty=penalty, class_weight='auto', dual=False) 
                                for C in C_values for penalty in  ['l1', 'l2']]))
                  for k in k_values])

# Take a look
print [l for l in anova_svms.walk_leaves()]

## k and C selection based on CV
anova_svms_auto = CVBestSearchRefit(anova_svms)

#anova_svm_all = Methods(anova_svm, anova_svm_cv)
cv = CV(anova_svms_auto, n_folds=n_folds)
time_fit_predict = time.time()
cv.run(X=X, y=y)
print time.time() - time_fit_predict
print cv.reduce()

# Re-fit on all data. Warning: biased !!!
anova_svms_auto.run(X=X, y=y)

print anova_svms_auto.best_params
print "Features selected by univariate filter"
selected_features = imaging_variables[anova_svms_auto.refited.estimator.get_support()]

print "Features selected weights"
d = dict(var = selected_features,
svm_weights_l1 = anova_svms_auto.refited.children[0].children[0].estimator.coef_.ravel())
print pd.DataFrame(d).to_string()


##############################################################################
## Use multi-process
from epac.map_reduce.engine import LocalEngine
time_fit_predict = time.time()
local_engine = LocalEngine(tree_root=cv, num_processes=4)
wf = local_engine.run(X=X, y=y)
print time.time() - time_fit_predict

print wf.reduce()

