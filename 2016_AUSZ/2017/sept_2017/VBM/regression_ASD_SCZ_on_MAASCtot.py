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
from sklearn.metrics import r2_score
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


WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_ASD_SCZ_MAASCtot'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/MASCtot.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/data_ASD_SCZ_only/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/mask.nii'



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
DX = np.load(INPUT_DATA_DX)

#Remove nan lines
X = X[np.logical_not(np.isnan(y)).ravel(),:]
DX = DX[np.logical_not(np.isnan(y))]
y = y[np.logical_not(np.isnan(y))]



lr = linear_model.Ridge(fit_intercept = True,alpha=100)
pred = sklearn.cross_validation.cross_val_predict(lr,X[:,3:] , y, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
r_squared = r2_score(y, pred)
print(r_squared)
np.corrcoef(y, pred)


scores = sklearn.cross_validation.cross_val_score(lr, X[:,3:], y, cv=5, scoring='r2')
print (scores)





lr = linear_model.Ridge(fit_intercept = True,alpha=1000)
lr.fit(X[50:,3:],y[50:])
plt.plot(y[:50],lr.predict(X[:50,3:]),'o')
plt.plot(y[50:],lr.predict(X[50:,3:]),'o')
r_squared = r2_score(y[:50],lr.predict(X[:50,3:]))
print(r_squared)
np.corrcoef(y[:50],lr.predict(X[:50,3:]))





plt.plot(y, pred, 'o', label='original data')
plt.plot(y, intercept + slope*y, 'r', label='fitted line')
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend()
plt.show()

plt.figure()
plt.grid()
plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)
plt.plot(y[DX==1], pred[DX==1], 'o',label = "ASD")
plt.plot(y[DX==3], pred[DX==3], 'o',label = "SCZ")
plt.plot(y, intercept + slope*y, 'r',color = "black")
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend(loc = "bottom")

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_ASD_SCZ_MAASCtot/regression_MAASC.png")

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

