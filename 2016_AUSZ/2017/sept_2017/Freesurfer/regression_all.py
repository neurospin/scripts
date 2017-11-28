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


WD = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/linear_regression_all'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/mask.nii'



##################################################################################

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
beta = beta[penalty_start:]

np.save(os.path.join(WD,"weight_map.npy"),beta)
