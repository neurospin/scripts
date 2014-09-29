# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:01:28 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

#import pickle
import numpy as np
import pandas as pd
#from joblib import Parallel, delayed
import pylab as plt

#import sklearn.cross_validation
#import sklearn.linear_model
#import sklearn.linear_model.coordinate_descent
import parsimony.functions.nesterov.tv as tv

## GLOBALS ==================================================================
BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"
INPUT_PATH = os.path.join(BASE_PATH, "tv", "split")

INPUT_X_TRAIN_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.train.center.npy")
INPUT_X_TEST_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.test.center.npy")
INPUT_Y_TRAIN_FILE = os.path.join(BASE_PATH, "y_CTL_AD.train.npy")
INPUT_Y_TEST_FILE = os.path.join(BASE_PATH, "y_CTL_AD.test.npy")
y = np.load(INPUT_Y_TEST_FILE)

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif_AD-CTL")
sys.path.append(SRC_PATH)
import utils_proj_classif


l2     = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-1.000-0.000-0.00")) # l2
l1     = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-0.000-1.000-0.00")) # ll
l1l2   = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-0.900-0.100-0.00"))# l1+l2
l2tv   = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-0.100-0.000-0.90")) # l2tv
l1tv   = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-0.100-0.000-0.90")) # l1tv
l1l2tv = utils_proj_classif.load(os.path.join(INPUT_PATH, "1.00-0.100-0.100-0.80")) # l1l2tv


comp = [
["l2", "l2tv"] + list(utils_proj_classif.mcnemar_test_prediction(y_pred1=l2['y_pred_tv'], y_pred2=l2tv['y_pred_tv']
, y_true=y)),

["l1", "l1tv"] + list(utils_proj_classif.mcnemar_test_prediction(y_pred1=l1['y_pred_tv'], y_pred2=l1tv['y_pred_tv']
, y_true=y)),

["l1l2", "l1l2tv"] + list(utils_proj_classif.mcnemar_test_prediction(y_pred1=l1l2['y_pred_tv'], y_pred2=l1l2tv['y_pred_tv']
, y_true=y))
]

pd.DataFrame(comp, columns=["method1", "method2", "pval_chi2", "pval_binom"])

"""
  method1 method2  pval_chi2  pval_binom
0      l2    l2tv   0.080856    0.156708
1      l1    l1tv   0.000204    0.000284
2    l1l2  l1l2tv   0.844519    1.000000
"""