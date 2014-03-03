# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:23:25 2013

@author: edouard.duchesnay@cea.fr
"""

import os.path
import sys
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC as SVM
#from sklearn.linear_model import LogisticRegression
#from sklearn import preprocessing
#from sklearn.feature_selection import SelectKBest
import epac
from epac import CV, Methods, Pipe, CVBestSearchRefit

WD = "/home/edouard/data/2013_predict-transition-caarms"
SRC = "/home/edouard/git/scripts/2013_predict-transition-caarms"

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)
import IO

################################################################################
## Dataset: CARMS + PAS + CANABIS
## Algo: SVM L1

Xd, yd = IO.read_Xy(WD=WD)
Xd.PAS2gr[Xd.PAS2gr==1] = -1
Xd.PAS2gr[Xd.PAS2gr==2] = 1
Xd.CB_EXPO[Xd.CB_EXPO==0] = -1

X = np.asarray(Xd)
y = np.asarray(yd)

C_values = [0.01, 0.05, .1, .5, 1, 5, 10]

# SVM L1
# ======

svms = Methods(*[SVM(dual=False, class_weight='auto', penalty="l1", C=C)  for C in C_values])

cv = CV(svms, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
#print cv_results

epac.export_csv(cv, cv_results, os.path.join(WD, "results", "cv10_caarms+pas+canabis_svmsl1.csv"))

# SVM L1 with CVBestSearchRefit
# =============================

svms_cv = CVBestSearchRefit(svms, n_folds=10, cv_type="stratified")
cv = CV(svms_cv, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results

#[{'key': CVBestSearchRefit, 'y/test/score_f1': [ 0.82352941  0.7       ], 'y/test/recall_pvalues': [ 0.01086887  0.06790736], 'y/test/score_precision': [ 0.77777778  0.77777778], 'y/test/recall_mean_pvalue': 0.0191572904587, 'y/test/score_recall': [ 0.875       0.63636364], 'y/test/score_accuracy': 0.777777777778, 'y/test/score_recall_mean': 0.755681818182}])

#
#Parmis les 27 11 ont fait la transition et 16 ne l'on pas faite
#- Sensibilité (Taux de detection de les transitions)
#63.63 % soit 7 / 11 (p = 0.067)
#
#- Spécificité (Taux de detection de ceux qui n'ont pas transité ou 1 - Faux positifs)
#87.5 % soit 14 / 16 (p = 0.01)
#
#Nous avons un taux de bonne classification moyen de 77 %
#
svms_cv.run(X=X, y=y)
#{'best_params': [{'C': 0.05, 'name': 'LinearSVC'}],
# 'y/pred': array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
#       0, 0, 0, 0]),
# 'y/true': array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
#       0, 0, 0, 0])}

svm = SVM(dual=False, class_weight='auto', penalty="l1", C=.05)
svm.fit(X, y)
svm.coef_
len(svm.coef_.squeeze())
# 27
coef = svm.coef_.squeeze()
print pd.DataFrame(dict(var=Xd.columns[coef !=0], coef=
coef[coef !=0]))
#       coef   var
#0 -0.084406  @4.3
#1  0.138804  @5.4
#2 -0.113875  @7.4
#3  0.063419  @7.6
#4  0.011630  @7.7


################################################################################
## Dataset: CARMS + PAS + CANABIS
## Algo: anova(P<0.05) SVM L1
import mylib


Xd, yd = IO.read_Xy(WD=WD)
Xd.PAS2gr[Xd.PAS2gr==1] = -1
Xd.PAS2gr[Xd.PAS2gr==2] = 1
Xd.CB_EXPO[Xd.CB_EXPO==0] = -1

X = np.asarray(Xd)
y = np.asarray(yd)

#k_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 27]
C_values = [0.01, 0.05, .1, .5, 1, 5, 10, 100, 1000]

# anova + SVM L1
# ==============

anova_svms = Pipe(mylib.SelectPvalue(alpha=1e-1),
                  Methods(*[SVM(C=C, penalty="l1", class_weight='auto', dual=False) for C in C_values]))


# P<0.05
cv = CV(anova_svms, cv_type="stratified", n_folds=10)
a=cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results

epac.export_csv(cv, cv_results, os.path.join(WD, "results", "cv10_caarms+pas+canabis_anova(p<0.05)_svmsl1.csv"))
# recall_mean: 67%

# Kbest
from sklearn.feature_selection import SelectKBest
anova_svms = Pipe(SelectKBest(k=5),
                  Methods(*[SVM(C=C, penalty="l2", class_weight='auto', dual=False) for C in C_values]))
cv = CV(anova_svms, cv_type="stratified", n_folds=10)
a=cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results

epac.export_csv(cv, cv_results, os.path.join(WD, "results", "cv10_caarms+pas+canabis_anova(k=5)_svmsl1.csv"))
# recall_mean: 59%
 
################################################################################
## Dataset: CARMS + PAS + CANABIS
## Algo: SVM L1

Xd, yd = IO.read_Xy(WD=WD)
Xd = Xd.drop(['PAS2gr', 'CB_EXPO'], 1)

X = np.asarray(Xd)
y = np.asarray(yd)

C_values = [0.01, 0.05, .1, .5, 1, 5, 10]

# SVM L1
# ======

# OK
svms = Methods(*[SVM(dual=False, class_weight='auto', penalty="l1", C=C)  for C in C_values])

svms = Methods(*[SVM(dual=False, class_weight='auto', loss='l1', penalty="l1", C=C)  for C in C_values])
svms_cv = CVBestSearchRefit(svms, n_folds=10, cv_type="stratified")
svms_all = Methods(svms, svms_cv)
cv = CV(svms_all, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results

# features
svms_cv.run(X=X, y=y)

svms_cv.refited.wrapped_node
#LinearSVC(C=0.05, class_weight='auto', dual=False, fit_intercept=True,
#     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l1',
#     random_state=None, tol=0.0001, verbose=0)
coef = svms_cv.refited.wrapped_node.coef_.squeeze()
print pd.DataFrame(dict(var=Xd.columns[coef !=0], coef=
coef[coef !=0]))

#       coef   var
#0 -0.084403  @4.3 anhédoni (symptome négatif)
#1  0.138800  @5.4 comportement agréssif dangereux
#2 -0.113874  @7.4 labilité de l'humeur
#3  0.063421  @7.6 trouble obsessionel et compulsif
#4  0.011630  @7.7 symptomes dissociatifs

svm = SVM(C=0.05, class_weight='auto', dual=False, fit_intercept=True, penalty='l1')
svm.fit(X, y)
coef = svm.coef_.squeeze()
print pd.DataFrame(dict(var=Xd.columns[coef !=0], coef=
coef[coef !=0]))
