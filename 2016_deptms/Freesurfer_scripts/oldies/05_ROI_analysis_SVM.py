#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:23:32 2016

@author: ad247405
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd


BASE_PATH="/neurospin/brainomics/2016_deptms"
INPUT_DATA_STATS = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/freesurfer_stats"
INPUT_DATA_Y = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/y.npy'
OUTPUT = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/results/svm_rois"
INPUT_CSV = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/population.csv"


pop = pd.read_csv(INPUT_CSV,delimiter=' ')
number_subjects = pop.shape[0]
NFOLDS_OUTER = 5
NFOLDS_INNER = 5


#############################################################################
## Create config file
y = np.load(INPUT_DATA_y)

cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
if cv_outer[0] is not None: # Make sure first fold is None
    cv_outer.insert(0, None)   
    null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
    cv_outer[0] = null_resampling
        
#     
import collections
cv = collections.OrderedDict()
for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
    if cv_outer_i == 0:
        cv["all/all"] = [tr_val, te]
     
    else:    
        cv["cv%02d/all" % (cv_outer_i -1)] = [tr_val, te]
        cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
        for cv_inner_i, (tr, val) in enumerate(cv_inner):
            cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
for k in cv:
    cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]

   
print(list(cv.keys()))  


rh_stats = pd.read_csv(os.path.join(INPUT_DATA_STATS,"aparc_thickness_rh_all.csv"),sep='\t')
lh_stats = pd.read_csv(os.path.join(INPUT_DATA_STATS,"aparc_thickness_lh_all.csv"),sep='\t')

rh_stats = np.asarray(rh_stats)
Xrh = rh_stats[:,1:].astype(float)
lh_stats = np.asarray(lh_stats)
Xlh = lh_stats[:,1:].astype(float)

X = np.hstack([Xrh,Xlh]).shape


### Run SVM on all features
n=0
list_predict=list()
list_true=list()
coef=np.zeros((24,sum(mask_bool)))
clf = svm.LinearSVC(C=10e-7,fit_intercept=False,class_weight='auto')

for i in range(1,24):
    test_bool=(subject==i)
    train_bool=(subject!=i)
    Xtest=T[test_bool,:]
    ytest=y[test_bool]
    Xtrain=T[train_bool,:]
    ytrain=y[train_bool]
    list_true.append(ytest.ravel())
    scaler = preprocessing.StandardScaler().fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest=scaler.transform(Xtest)
    clf.fit(Xtrain, ytrain.ravel())
    coef[n,:]=clf.coef_
    pred=(clf.predict(Xtest))
    list_predict.append(pred)
    print n 
    n=n+1 


class_weight="auto" # unbiased
    
    mask = np.ones(Xtr.shape[0], dtype=bool)
   
    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)    
    
    mod = svm.LinearSVC(C=c,fit_intercept=False,class_weight= class_weight)

    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.coef_,  mask=mask)