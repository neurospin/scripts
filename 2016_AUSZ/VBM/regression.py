#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:44:25 2017

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


BASE_PATH =  '/neurospin/brainomics/2016_AUSZ/results/VBM'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/linear_regression'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/results/VBM/data/X_patients_only.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/results/VBM/data/y_patients_only.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/results/VBM/data/mask.nii'



##################################################################################
babel_mask  = nibabel.load('/neurospin/brainomics/2016_AUSZ/results/VBM/data/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################


n_folds = 5
    
X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)


    
lr = linear_model.LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
pred = sklearn.cross_validation.cross_val_predict(lr,X , y, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)

plt.plot(y, pred, 'o', label='original data')
plt.plot(y, intercept + slope*y, 'r', label='fitted line')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend()
plt.show()


#Obain coef map
lr = linear_model.LinearRegression()
lr.fit(X,y)
beta = lr.coef_
np.save("/neurospin/brainomics/2016_AUSZ/results/VBM/linear_regression/beta",beta)
beta = beta[penalty_start:]

arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.0002, vmax = 0.0002)

