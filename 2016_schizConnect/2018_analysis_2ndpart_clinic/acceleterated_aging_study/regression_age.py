#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:23:54 2018

@author: ad247405
"""


import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel as nib
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn
from scipy import stats
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image
import array_utils
import scipy.stats
import pandas as pd
from sklearn import metrics

WD = '/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/accelerated_aging'

INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii'
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv"


##################################################################################
babel_mask  = nib.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 2
##################################################################################
U = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/\
struct_pca_0.1_0.1_0.1/X_test_transform.npz")["arr_0"]
U_con = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_con.npy")
U_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_scz.npy")

n_folds = 5
pop = pd.read_csv(INPUT_POPULATION )
X = np.load(INPUT_DATA_X)[:,penalty_start:]
y = np.load(INPUT_DATA_y)
age = pop["age"]
assert X.shape == (606, 125959)

X_con = X[y==0,:]
age_con = age[y==0]
X_scz = X[y==1,:]
age_scz = age[y==1]

lr = linear_model.LinearRegression(fit_intercept=True)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
pred = sklearn.cross_validation.cross_val_predict(lr,X_con , age_con, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(age_con, pred)

plt.plot(age_con, pred, 'o', label='original data')
plt.plot(age_con, intercept + slope*age_con, 'r', label='fitted line')
plt.xlabel("age")
plt.ylabel("Predicted age using MRI-based features")
plt.legend()
plt.xlim(10,70)
plt.ylim(10,70)
plt.show()

lr = linear_model.LinearRegression()
lr.fit(X_con,age_con)
pred_age_con = lr.predict(X_con)
pred_age_scz = lr.predict(X_scz)

plt.plot(age_scz, pred_age_scz, 'o', label='original data',color="r")
plt.plot(age_scz, intercept + slope*age_scz, 'r', label='fitted line',color="r")
plt.xlabel("age")
plt.ylabel("Predicted age using MRI-based features")
plt.legend()
plt.xlim(10,70)
plt.ylim(10,70)
plt.show()



lr = linear_model.LinearRegression()

pred = sklearn.cross_validation.cross_val_predict(lr,U_con , age_con, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(age_con, pred)
r_squared = metrics.r2_score(age_con, pred)
plt.plot(age_con, pred, 'o', label='original data')
plt.plot(age_con, intercept + slope*age_con, 'r', label='fitted line')
plt.xlabel("age")
plt.ylabel("Predicted age using MRI-based features")
plt.legend()
plt.xlim(10,70)
plt.ylim(10,70)
plt.show()


lr = linear_model.LinearRegression()
lr.fit(U_con,age_con)
pred_age_scz = lr.predict(U_scz)
slope, intercept, r_value, p_value, std_err = stats.linregress(age_scz, pred_age_scz)
r_squared = metrics.r2_score(age_scz, pred_age_scz)
lr.coef_



plt.plot(age_scz, pred_age_scz, 'o', label='original data',color="r")
plt.plot(age_scz, intercept + slope*age_scz, 'r', label='fitted line',color="r")
plt.xlabel("age")
plt.ylabel("Predicted age using MRI-based features")
plt.legend()
plt.xlim(10,70)
plt.ylim(10,70)
plt.show()





#Obain coef map
lr = linear_model.LinearRegression()
lr.fit(U_con,age_con)
beta = lr.coef_
beta = beta[penalty_start:]

arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nib.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nib.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

