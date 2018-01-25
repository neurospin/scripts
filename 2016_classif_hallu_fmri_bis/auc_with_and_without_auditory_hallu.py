#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:53:06 2017

@author: ad247405
"""


import nibabel as nibabel
import numpy as np
import os
import nilearn.signal
import nilearn.image
import re
import glob
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import random
import brainomics
import matplotlib.pyplot as plt



results_csv = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/\
multivariate_analysis/enettv/model_selection/results_dCV_recall_mean.xlsx"


results_by_cv = pd.read_excel(results_csv, sheetname=2)

true = list()
pred= list()
proba = list()


data_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/\
multivariate_analysis/enettv/model_selection/model_selectionCV"

cv_no_auditory = ["cv04","cv12","cv17","cv21","cv28"]
for cv in cv_no_auditory:

    param = results_by_cv[results_by_cv["fold"]==cv]["param_key"].values[0]
    os.path.join(data_path,cv,"refit",param)
    y_true = np.load(os.path.join(data_path,cv,"refit",param,"y_true.npz"))['arr_0']
    y_pred = np.load(os.path.join(data_path,cv,"refit",param,"y_pred.npz"))['arr_0']
    proba_pred = np.load(os.path.join(data_path,cv,"refit",param,"proba_pred.npz"))['arr_0']
    true.append(y_true.ravel())
    pred.append(y_pred.ravel())
    proba.append(proba_pred.ravel())

p, r, f, s = precision_recall_fscore_support(np.hstack(true), np.hstack(pred), average=None)
roc_auc_score(np.hstack(true), np.hstack(pred))
print (r.mean())
print (roc_auc_score(np.hstack(true), np.hstack(proba)))


true = list()
pred= list()
proba = list()

cv_auditory = ["cv00","cv01","cv02","cv03","cv05","cv06","cv07","cv08","cv09","cv10","cv11","cv12",\
                 "cv13","cv14","cv15","cv16","cv18","cv19","cv20","cv22","cv23",\
                  "cv24","cv25","cv26","cv27","cv29","cv30","cv31","cv32","cv33","cv34","cv35","cv36"]
for cv in cv_auditory:

    param = results_by_cv[results_by_cv["fold"]==cv]["param_key"].values[0]
    os.path.join(data_path,cv,"refit",param)
    y_true = np.load(os.path.join(data_path,cv,"refit",param,"y_true.npz"))['arr_0']
    y_pred = np.load(os.path.join(data_path,cv,"refit",param,"y_pred.npz"))['arr_0']
    proba_pred = np.load(os.path.join(data_path,cv,"refit",param,"proba_pred.npz"))['arr_0']
    true.append(y_true.ravel())
    pred.append(y_pred.ravel())
    proba.append(proba_pred.ravel())

p, r, f, s = precision_recall_fscore_support(np.hstack(true), np.hstack(pred), average=None)
print (r.mean())
print (roc_auc_score(np.hstack(true), np.hstack(proba)))


true = list()
pred= list()
proba = list()
cv_auditory = ["cv00","cv01","cv02","cv03","cv04","cv05","cv06","cv07","cv08","cv09","cv10","cv11","cv12",\
                 "cv13","cv14","cv15","cv16","cv17","cv18","cv19","cv20","cv21","cv22","cv23",\
                  "cv24","cv25","cv26","cv27","cv28","cv29","cv30","cv31","cv32","cv33","cv34","cv35","cv36"]
for cv in cv_auditory:

    param = results_by_cv[results_by_cv["fold"]==cv]["param_key"].values[0]
    os.path.join(data_path,cv,"refit",param)
    y_true = np.load(os.path.join(data_path,cv,"refit",param,"y_true.npz"))['arr_0']
    y_pred = np.load(os.path.join(data_path,cv,"refit",param,"y_pred.npz"))['arr_0']
    proba_pred = np.load(os.path.join(data_path,cv,"refit",param,"proba_pred.npz"))['arr_0']
    true.append(y_true.ravel())
    pred.append(y_pred.ravel())
    proba.append(proba_pred.ravel())

p, r, f, s = precision_recall_fscore_support(np.hstack(true), np.hstack(pred), average=None)
print (r.mean())
print (roc_auc_score(np.hstack(true), np.hstack(proba)))

