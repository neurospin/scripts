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


WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/linear_regression_patients_on_MAASC'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/X_patients.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/MASCtot_patients.npy'
INPUT_DATA_DX = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/y_patients.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/mask.npy'


n_folds = 5

X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)
DX = np.load(INPUT_DATA_DX).reshape(92,1)

#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
DX=DX[np.logical_not(np.isnan(y))].reshape(80)
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (80, 317104)

#lr = linear_model.LinearRegression(fit_intercept=False)
lr = linear_model.LinearRegression(fit_intercept=True)

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
pred = sklearn.cross_validation.cross_val_predict(lr,X , y, cv=n_folds)
slope, intercept, r_value, p_value, std_err = stats.linregress(y, pred)



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

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/linear_regression_patients_on_MAASC/regression_MAASC.png")

