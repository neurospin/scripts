#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:32:16 2017

@author: ad247405
"""


import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import svm
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt
from sklearn import preprocessing


#Schizconnect
INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_selected_ROIs"
X_scz = np.load(os.path.join(INPUT_DATA,"Xrois_volumes_mean_centered_by_site+cov.npy"))
y_scz = np.load(os.path.join(INPUT_DATA,"y.npy"))
features_scz = np.load(os.path.join(INPUT_DATA,"features.npy"))
assert X_scz.shape[1] == (49)

#PRAGUE subjects
INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer/data/data_ROIs"
X_pra = np.load(os.path.join(INPUT_DATA,"Xrois_volumes_mean_centered_by_site+cov.npy"))
y_pra = np.load(os.path.join(INPUT_DATA,"y.npy"))
assert X_pra.shape[1] == (49)


scaler = preprocessing.StandardScaler().fit(X_scz)
X_scz = scaler.transform(X_scz)
X_pra = scaler.transform(X_pra)


mod = svm.LinearSVC(C =0.01,fit_intercept=False,class_weight= "auto")


mod.fit(X_scz,y_scz)
y_pred_pra = mod.predict(X_pra)
y_proba_pred_pra = mod.decision_function(X_pra)

p, r, f, s = precision_recall_fscore_support(y_pra, y_pred_pra, average=None)
auc = roc_auc_score(y_pra, y_proba_pred_pra)

print("######################################")
print("Classification performance on PRAGUE dataset:")
print("Balanced accuracy : " + str(r.mean()))
print("Spe and Sen : " + str(r[0]) + " " + str(r[1]))
print("AUC : " + str(auc))
print("######################################")

######################################
#Classification performance on PRAGUE dataset:
#Balanced accuracy : 0.581746031746
#Spe and Sen : 0.377777777778 0.785714285714
#AUC : 0.751058201058
#######################################8 paranoid SCZ and 41 Acute polymorphic psychotic disorder with symptoms of schizophrenia