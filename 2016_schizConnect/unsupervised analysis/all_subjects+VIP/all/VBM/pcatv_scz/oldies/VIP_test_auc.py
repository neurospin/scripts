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

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/data_by_site/VIP/y.npy")

cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=5, random_state=42)]


results_csv = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/\
enetall_VIP/enetall_VIP_dcv.xlsx"


results_by_cv = pd.read_excel(results_csv, sheetname=2)

true = list()
pred= list()
proba = list()


data_path = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/enetall_VIP/5cv"

cv_range = ["cv00","cv01","cv02","cv03","cv04"]
for cv in cv_range:

    param = results_by_cv[results_by_cv["fold"]==cv]["key"].values[0]
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

panss_neg_score = pop_vip["PANSS_NEGATIVE"]
panss_pos_score = pop_vip["PANSS_POSITIVE"]

true_scz = list()
pred_scz = list()
panss_neg_ok = list()
panss_neg_no = list()
panss_pos_ok = list()
panss_pos_no = list()

for j in range(5):
    for i in range(len(true[j])):
        if(true[j][i]==1):
            if (pred[j][i] == 1):
                print("ok")
                true_scz.append(1)
                pred_scz.append(1)
                panss_neg_ok.append(panss_neg_score[cv_outer[j][1][i]])
                panss_pos_ok.append(panss_pos_score[cv_outer[j][1][i]])
            if (pred[j][i] == 0):
                print("NO")
                true_scz.append(1)
                pred_scz.append(0)
                panss_pos_no.append(panss_pos_score[cv_outer[j][1][i]])
                panss_neg_no.append(panss_neg_score[cv_outer[j][1][i]])

p, r, f, s = precision_recall_fscore_support(np.hstack(true_scz), np.hstack(pred_scz), average=None)
print (r[1])

np.nanmean(panss_neg_ok)
np.nanmean(panss_neg_no)

np.nanmean(panss_pos_ok)
np.nanmean(panss_pos_no)


pop_vip = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv")
age = pop_vip["age"]

panss_pos = pop_vip["PANSS_POSITIVE"]
panss_galp = pop_vip["PANSS_GALPSYCHOPAT"]
panss_comp = pop_vip["PANSS_COMPOSITE"]
cdss = pop_vip["CDSS_Total"]
fast = pop_vip["FAST_TOT"]
