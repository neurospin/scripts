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
from sklearn.cross_validation import KFold
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
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from sklearn.metrics import r2_score

WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_patients_MAASCtot'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/X_patients_only.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/MASCtot_patients_only.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/mask.nii'
INPUT_DATA_MASCexc = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/MASCexc_patients_only.npy'
INPUT_DATA_MASCless = '/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/data/data_with_intercept/MASCless_patients_only.npy'



##################################################################################
babel_mask  = nib.load(INPUT_MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################


n_folds = 5

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
DX = np.load(INPUT_DATA_DX)
MASCexc = np.load(INPUT_DATA_MASCexc)
MASCless = np.load(INPUT_DATA_MASCless)



X = X[np.logical_not(np.isnan(y)).ravel(),:]
DX = DX[np.logical_not(np.isnan(y))]
MASCexc = MASCexc[np.logical_not(np.isnan(y))]
MASCless = MASCless[np.logical_not(np.isnan(y))]
y = y[np.logical_not(np.isnan(y))]


#lr = linear_model.Ridge(fit_intercept=True)
#
#scaler = preprocessing.StandardScaler().fit(X[:,penalty_start:])
#X[:,penalty_start:] = scaler.transform(X[:,penalty_start:])
#
#
#pred = sklearn.cross_validation.cross_val_predict(lr,X , y, cv = 5)
#slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
#r_squared = r2_score(y, pred)
#np.corrcoef(y, pred)
#print(r_squared)



## Create config file
cv_outer = [[tr, te] for tr,te in KFold(n=len(y), n_folds=5, random_state=42)]


lr = linear_model.Ridge(fit_intercept = True,alpha=1e4)
lr.fit(X[cv_outer[0][0],3:],y[cv_outer[0][0]])
plt.plot(y[cv_outer[0][0]],lr.predict(X[cv_outer[0][0],3:]),'o')
plt.plot(y[cv_outer[0][1]],lr.predict(X[cv_outer[0][1],3:]),'o')
r_squared = r2_score(y[cv_outer[0][1]],lr.predict(X[cv_outer[0][1],3:]))
print(r_squared)
np.corrcoef(y[cv_outer[0][1]],lr.predict(X[cv_outer[0][1],3:]))



scaler = preprocessing.StandardScaler().fit(X[cv_outer[1][0],3:])
X[:,3:] = scaler.transform(X[:,3:])
lr = linear_model.Ridge(fit_intercept = True,alpha=1e4)
lr.fit(X[cv_outer[1][0],3:],y[cv_outer[1][0]])
plt.plot(y[cv_outer[1][0]],lr.predict(X[cv_outer[1][0],3:]),'o')
plt.plot(y[cv_outer[1][1]],lr.predict(X[cv_outer[1][1],3:]),'o')
r_squared = r2_score(y[cv_outer[1][1]],lr.predict(X[cv_outer[1][1],3:]))
print(r_squared)
np.corrcoef(y[cv_outer[1][1]],lr.predict(X[cv_outer[1][1],3:]))


scaler = preprocessing.StandardScaler().fit(X[cv_outer[2][0],3:])
X[:,3:] = scaler.transform(X[:,3:])
lr = linear_model.Ridge(fit_intercept = True,alpha=1e4)
lr.fit(X[cv_outer[2][0],3:],y[cv_outer[2][0]])
plt.plot(y[cv_outer[2][0]],lr.predict(X[cv_outer[2][0],3:]),'o')
plt.plot(y[cv_outer[2][1]],lr.predict(X[cv_outer[2][1],3:]),'o')
r_squared = r2_score(y[cv_outer[2][1]],lr.predict(X[cv_outer[2][1],3:]))
print(r_squared)
np.corrcoef(y[cv_outer[2][1]],lr.predict(X[cv_outer[2][1],3:]))

scaler = preprocessing.StandardScaler().fit(X[cv_outer[3][0],3:])
X[:,3:] = scaler.transform(X[:,3:])
lr = linear_model.Ridge(fit_intercept = True,alpha=1e4)
lr.fit(X[cv_outer[3][0],3:],y[cv_outer[3][0]])
plt.plot(y[cv_outer[3][0]],lr.predict(X[cv_outer[3][0],3:]),'o')
plt.plot(y[cv_outer[3][1]],lr.predict(X[cv_outer[3][1],3:]),'o')
r_squared = r2_score(y[cv_outer[3][1]],lr.predict(X[cv_outer[3][1],3:]))
print(r_squared)
np.corrcoef(y[cv_outer[3][1]],lr.predict(X[cv_outer[3][1],3:]))

scaler = preprocessing.StandardScaler().fit(X[cv_outer[4][0],3:])
X[:,3:] = scaler.transform(X[:,3:])
lr = linear_model.Ridge(fit_intercept = True,alpha=1e4)
lr.fit(X[cv_outer[4][0],3:],y[cv_outer[4][0]])
plt.plot(y[cv_outer[4][0]],lr.predict(X[cv_outer[4][0],3:]),'o')
plt.plot(y[cv_outer[4][1]],lr.predict(X[cv_outer[4][1],3:]),'o')
r_squared = r2_score(y[cv_outer[4][1]],lr.predict(X[cv_outer[4][1],3:]))
print(r_squared)
np.corrcoef(y[cv_outer[4][1]],lr.predict(X[cv_outer[4][1],3:]))








plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)
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
plt.plot(y[DX==2], pred[DX==2], 'o',label = "SCZ-ASD")
plt.plot(y[DX==3], pred[DX==3], 'o',label = "SCZ")
plt.plot(y, intercept + slope*y, 'r',color = "black")
plt.xlabel("MAASC score")
plt.ylabel("Predicted score using MRI-based features")
plt.legend(loc = "bottom left")

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/linear_regression_patients_MAASCtot/regression_MAASC.png")

#
##
##
##mod = estimators.RidgeRegression(0.1,penalty_start = 3)
##mod.fit(X, y)
##mod.predict(X)
##Obain coef map
#lr = linear_model.LinearRegression()
#
#lr.fit(X,y)
#lr.predict(X)
#beta = lr.coef_
#
#arr = np.zeros(mask_bool.shape);
#arr[mask_bool] = beta.ravel()
#out_im = nib.Nifti1Image(arr, affine=babel_mask.get_affine())
#filename = os.path.join(WD,"weight_map.nii.gz")
#out_im.to_filename(filename)
#beta = nib.load(filename).get_data()
#
#beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
#
#
##ElasticNet model
#lr = linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
## cross_val_predict returns an array of the same size as `y` where each entry
## is a prediction obtained by cross validation:
#pred = sklearn.cross_validation.cross_val_predict(lr,X , y, cv=n_folds)
#slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
#
#plt.figure()
#plt.grid()
#plt.title("R2 = %.02f and p = %.01e" % (r_value,p_value),fontsize=12)
#plt.plot(y[DX==1], pred[DX==1], 'o',label = "ASD")
#plt.plot(y[DX==2], pred[DX==2], 'o',label = "SCZ-ASD")
#plt.plot(y[DX==3], pred[DX==3], 'o',label = "SCZ")
#plt.plot(y, intercept + slope*y, 'r',color = "black")
#plt.xlabel("MAASC score")
#plt.ylabel("Predicted score using MRI-based features")
#
#
#
#mod = linear_model.LinearRegression()
#mod.fit(Xtr, ytr.ravel())
#y_pred = mod.predict(Xte)
#
##Test some function os sklearn to check consistency of resutls
#
##lr = linear_model.LinearRegression()
##cv = sklearn.cross_validation.KFold(n=len(y),n_folds=5)
##pred = sklearn.cross_validation.cross_val_predict(lr, X, y, cv = cv)
##slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
##
##import sklearn.pipeline
##lr = linear_model.LinearRegression()
##scalar = preprocessing.StandardScaler()
##clf = linear_model.LinearRegression()
##pip= sklearn.pipeline.Pipeline([('transformer', scalar), ('estimator', lr)])
##cv = KFold(n=5)
##pred = sklearn.cross_validation.cross_val_predict(pip, X, y, cv = cv)
##slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
##
##import sklearn.pipeline
##lr = linear_model.LinearRegression()
##scalar = preprocessing.StandardScaler()
##clf = linear_model.LinearRegression()
##pip= sklearn.pipeline.Pipeline([('estimator', lr)])
##cv = sklearn.cross_validation.KFold(n=len(y),n_folds=5)
##pred = sklearn.cross_validation.cross_val_predict(pip, X, y, cv = cv)
##slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)
#
