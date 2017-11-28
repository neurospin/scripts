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
import scipy.stats


WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_all_MAASCtot'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_for_regression_all/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_for_regression_all/MASCtot.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_for_regression_all/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_for_regression_all/mask.nii'



##################################################################################
babel_mask  = nib.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 2
##################################################################################


n_folds = 5

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
DX = np.load(INPUT_DATA_DX).reshape(123,1)

assert X.shape == (123, 261212)

#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
DX=DX[np.logical_not(np.isnan(y))]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (110, 261212)

#0: controls
#1: SCZ
#2: SCZ-ASD
#3: ASD
#Significant differences between MAASC score of ADS, SCZ and controls
scipy.stats.f_oneway(y[DX==0],y[DX==1],y[DX==2],y[DX==3])
scipy.stats.f_oneway(y[DX==0],y[DX==2],y[DX==3])

scipy.stats.ttest_ind(y[DX==1],y[DX==3])
scipy.stats.ttest_ind(y[DX==0],y[DX==3])
scipy.stats.ttest_ind(y[DX==0],y[DX==1])


lr = linear_model.LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
pred = sklearn.cross_validation.cross_val_predict(lr,X , y, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)

plt.plot(y, pred, 'o', label='original data')
plt.plot(y, intercept + slope*y, 'r', label='fitted line')
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)
plt.plot(y[DX==0], pred[DX==0], 'o',label = "controls")
plt.plot(y[DX==1], pred[DX==1], 'o',label = "SCZ")
plt.plot(y[DX==2], pred[DX==2], 'o',label = "SCZ-ASD")
plt.plot(y[DX==3], pred[DX==3], 'o',label = "ASD")
plt.plot(y, intercept + slope*y, 'r',color = "black")
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend(loc = "bottom left")
plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_all_MAASCtot/regression_MAASC.png")

#Obain coef map
lr = linear_model.LinearRegression()
lr.fit(X,y)
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

