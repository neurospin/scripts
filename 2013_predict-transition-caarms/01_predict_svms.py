# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:23:25 2013

@author: edouard
"""

import os.path
import sys
import numpy as np
from sklearn.svm import LinearSVC as SVM
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
import epac
from epac import CV, Methods, Pipe, CVBestSearchRefit

WD = "/home/edouard/data/2013_predict-transition-caarms"
SRC = "/home/edouard/git/scripts/2013_predict-transition-caarms"

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)

import IO
Xd, yd = IO.read_Xy(WD=WD)
X = np.asarray(Xd)
y = np.asarray(yd)

# DO NOT scale
#X -= X.mean(axis=0)
#X /= X.std(axis=0)


#k_values = [1, 2, 3, 4, 5, 10, 15, 20, 25, 27]
C_values = [0.01, 0.05, .1, .5, 1, 5, 10]

# SVM L1
# ======

svms = Methods(*[SVM(dual=False, class_weight='auto', penalty="l1", C=C)  for C in C_values])
       
#
#anova_svms = Methods(*[Pipe(SelectKBest(k=k),       #preprocessing.StandardScaler(),
#                            Methods(*[SVM(C=C, penalty=penalty, class_weight='auto', dual=False) for C in C_values for penalty in  ['l1', 'l2']])) for k in k_values])


cv = CV(svms, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
#print cv_results

epac.export_csv(cv, cv_results, os.path.join(WD, "cv10_svmsl1.csv"))

# SVM L1 with CVBestSearchRefit
# =============================

svms_cv = CVBestSearchRefit(svms, n_folds=10)
cv = CV(svms_cv, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results
#[{'key': CVBestSearchRefit, 'y/test/score_f1': [ 0.84848485  0.76190476], 'y/test/recall_pvalues': [ 0.01086887  0.03000108], 'y/test/score_precision': [ 0.82352941  0.8       ], 'y/test/recall_mean_pvalue': 0.00592461228371, 'y/test/score_recall': [ 0.875       0.72727273], 'y/test/score_accuracy': 0.814814814815, 'y/test/score_recall_mean': 0.801136363636}])
#
#Parmis les 27 11 ont fait la transition et 16 ne l'on pas faite
#- Sensibilité (Taux de detection de les transitions)
#72.72 % soit 8 / 11 (p = 0.03)
#
#- Spécificité (Taux de detection de ceux qui n'ont pas transité ou 1 - Faux positifs)
#87.5 % soit 14 / 16 (p = 0.01)
#
#Nous avons un taux de bonne classification moyen de 81.4 %
#
#Voici les items de la CAARMS qui interviennnent:
#[['@4.3', -0.084408961133411356],
# ['@5.4', 0.13881360208187149],
# ['@7.4', -0.11387844064581529],
# ['@7.6', 0.0634145029598185],
# ['@7.7', 0.011629021906204358]]
#Le chiffre à coté donne le poids (plus il est grand EN VALEUR ABSOLUE) plus l'item participe à la prédiction
#
#J'ai essayé de normaliser le Pre-Morbid Adjustment scale (PAS2gr) et l'exposition au canabis (CB_EXPO). Cela dégrade considérablement les résultats. Je ne sais pas encore pourquoi et la j'ai envie de te rejoindre dans le lit...



svms_cv.run(X=X, y=y)
#{'best_params': [{'C': 0.05, 'name': 'LinearSVC'}],
# 'y/pred': array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
#       0, 0, 0, 0]),
# 'y/true': array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
#       0, 0, 0, 0])}
cv_results = cv.reduce()
print cv_results


# LogisticRegression L1
# =====================

lr = Methods(*[LogisticRegression(dual=False, class_weight='auto', penalty="l1", C=C)  for C in C_values])

cv = CV(lr, cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
epac.export_csv(cv, cv_results, os.path.join(WD, "cv10_logregl1.csv"))

##########################################################################
loo = CV(svms, cv_type="loo")
loo.run(X=X, y=y)
loo_results = loo.reduce()
epac.export_csv(loo, loo_results, os.path.join(WD, "loo_svmsl1.csv"))

######################################################################
# LinearSVC(penalty=l1,C=0.05)
# recall: [ 0.9375      0.81818182]		[ 0.00196404  0.00500466]
# mean recall: 0.8778409091	0.0003107488

svm = SVM(dual=False, penalty="l1", C=0.05, class_weight='auto')
svm.fit(X, y)
svm.coef_[0]

[[Xd.columns[i], svm.coef_[0][i]] for i in xrange(len(svm.coef_[0])) if svm.coef_[0][i] != 0]

[['@4.3', -0.084408961133411356],
 ['@5.4', 0.13881360208187149],
 ['@7.4', -0.11387844064581529],
 ['@7.6', 0.0634145029598185],
 ['@7.7', 0.011629021906204358]]
 
######################################################################
class SVML1MinPena:
    def __init__(self, c_values, n_non_null):
         self.c_values = c_values
         self.n_non_null = n_non_null
    def fit(self, X, y):
        for C in self.c_values:
            svm = SVM(dual=False, penalty="l1", C=C, class_weight='auto')
            svm.fit(X, y)
            if np.sum(svm.coef_!=0) >= self.n_non_null:
                self.svm = svm
                self.coef_ = self.svm.coef_
                self.C = C
                break
        return self
    def predict(self, X):
        print self.C
        return self.svm.predict(X)

c_values = np.arange(0.001, 10, .001)
svm_minpena = SVML1MinPena(c_values, 5)
svm_minpena.fit(X, y)
svm_minpena.coef_
svm_minpena.C


cv = CV(SVML1MinPena(c_values,5), cv_type="stratified", n_folds=10)
cv.run(X=X, y=y)
cv_results = cv.reduce()
print cv_results
