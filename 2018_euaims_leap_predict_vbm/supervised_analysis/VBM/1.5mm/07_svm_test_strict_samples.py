#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:48:23 2018

@author: ad247405
"""

import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn import grid_search, metrics
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import scipy
from scipy import stats
from sklearn.preprocessing import StandardScaler
import sklearn

BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
DATA_PATH = "/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data"


pop = pd.read_csv(INPUT_CSV)
age = pop["age"].values
sex = pop["sex_num"].values

categories = pop["schedule_enrol"].values
site = pop["site"].values



# ADOS TOTAL SCORE PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))


clf = svm.LinearSVC(C=1,fit_intercept=False,class_weight='auto')
pred = sklearn.cross_validation.cross_val_predict(clf,X ,y.ravel(), cv=5)

recall_scores = recall_score(y,pred,pos_label=None, average=None,labels=[0,1])
acc=metrics.accuracy_score(y,pred)
print(acc)
print (recall_scores)

###############################################################################




###############################################################################
# ADOS TOTAL SCORE PREDICTION by site
###############################################################################


for i in range(1,7):
    X = np.load(os.path.join(DATA_PATH,"X.npy"))
    y = np.load(os.path.join(DATA_PATH,"y.npy"))

    X = X[site==i,:]
    y = y[site==i]

    clf = svm.LinearSVC(C=1,fit_intercept=False,class_weight='auto')
    pred = sklearn.cross_validation.cross_val_predict(clf,X ,y.ravel(), cv=5)
    recall_scores = recall_score(y,pred,pos_label=None, average=None,labels=[0,1])
    acc=metrics.accuracy_score(y,pred)
    print(acc)
    print (recall_scores)




###############################################################################


# ADOS TOTAL SCORE PREDICTION
###############################################################################
X = np.load(os.path.join(DATA_PATH,"X.npy"))
y = np.load(os.path.join(DATA_PATH,"y.npy"))

X = X[categories==3,:]
y = y[categories==3]


clf = svm.LinearSVC(C=1,fit_intercept=False,class_weight='auto')
pred = sklearn.cross_validation.cross_val_predict(clf,X ,y.ravel(), cv=5)

recall_scores = recall_score(y,pred,pos_label=None, average=None,labels=[0,1])
acc=metrics.accuracy_score(y,pred)
print(acc)
print (recall_scores)
###############################################################################

