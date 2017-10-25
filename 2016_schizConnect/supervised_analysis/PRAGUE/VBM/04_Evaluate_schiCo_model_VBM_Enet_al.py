#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:32:16 2017

@author: ad247405
"""

import os
import json
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.linalgs import LinearOperatorNesterov
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from statsmodels.stats.inter_rater import fleiss_kappa

penalty_start = 2
class_weight = "auto"
#Schizconnect
INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mean_centered_by_site_all"
X_scz = np.load(os.path.join(INPUT_DATA,"X.npy"))
y_scz = np.load(os.path.join(INPUT_DATA,"y.npy"))
assert  X_scz.shape[1] == 125961

import scipy.sparse as sparse
Atv = LinearOperatorNesterov(filename=(os.path.join(INPUT_DATA,"Atv.npz")))
Agn = sparse.vstack(Atv)
Agn.singular_values = Atv.get_singular_values()

#PRAGUE subjects
INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM/data"
X_pra = np.load(os.path.join(INPUT_DATA,"X.npy"))
y_pra = np.load(os.path.join(INPUT_DATA,"y.npy"))
assert X_pra.shape[1] == 125961

scaler = preprocessing.StandardScaler().fit(X_scz)
X_scz = scaler.transform(X_scz)
X_pra = scaler.transform(X_pra)

##'enettv':
tvratio=0.01
l1l2ratio = 0.1
alpha = 0.01
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)

conesta = algorithms.proximal.CONESTA(max_iter=10000)
mod = estimators.LogisticRegressionL1L2TV(l1, l2, tv,Atv,\
algorithm=conesta, class_weight=class_weight, penalty_start=penalty_start)

#
#'enetgn':
tvratio=0.1
l1l2ratio = 0.1
alpha = 0.1
tv = alpha * tvratio
l1 = alpha * float(1 - tv) * l1l2ratio
l2 = alpha * float(1 - tv) * (1- l1l2ratio)
fista = algorithms.proximal.FISTA(max_iter=5000)
mod = estimators.LogisticRegressionL1L2GraphNet(l1, l2, tv,Agn,
algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)

#algo == 'enet':
fista = algorithms.proximal.FISTA(max_iter=5000)
mod = estimators.ElasticNetLogisticRegression(0.1,0.1,
algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)



mod.fit(X_scz,y_scz)
y_pred_pra = mod.predict(X_pra)
y_proba_pred_pra = mod.predict_probability(X_pra)

p, r, f, s = precision_recall_fscore_support(y_pra, y_pred_pra, average=None)
auc = roc_auc_score(y_pra, y_proba_pred_pra)

print("######################################")
print("Classification performance on PRAGUE dataset:")
print("Balanced accuracy : " + str(r.mean()))
print("Spe and Sen : " + str(r[0]) + " " + str(r[1]))
print("AUC : " + str(auc))
print("######################################")


#ENET : 0.1/0.1/0.1
######################################
#Classification performance on PRAGUE dataset:
#Balanced accuracy : 0.671059431525
#Spe and Sen : 0.644444444444 0.697674418605
#AUC : 0.727131782946
######################################