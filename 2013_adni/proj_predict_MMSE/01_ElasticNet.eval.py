# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:44:49 2014

@author: md238665
"""


import os

import numpy as np
import pandas as pd

import sklearn.cross_validation
import sklearn.linear_model
import sklearn.linear_model.coordinate_descent
import sklearn.metrics

import proj_predict_MMSE_config

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"

INPUT_X_CENTER_FILE = os.path.join(BASE_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(BASE_PATH, "y.center.npy")
INPUT_PATH = os.path.join(BASE_PATH, "ElasticNet")
INPUT_ALL_GLOBAL_PENALIZATION = os.path.join(INPUT_PATH, "alphas.npy")

OUTPUT_PATH = INPUT_PATH

OUTPUT_RSQUARED = os.path.join(OUTPUT_PATH, "r_squared.npy")
OUTPUT_OPT_L1_RATIO = os.path.join(OUTPUT_PATH, "opt_l1_ratio.npy")
OUTPUT_ALL_GLOBAL_PENALIZATION = os.path.join(OUTPUT_PATH, "alphas.npy")
# Will be used for TV penalized
OUTPUT_OPT_GLOBAL_PENALIZATION = os.path.join(OUTPUT_PATH, "opt_alpha.npy")

#############
# Load data #
#############

X = np.load(INPUT_X_CENTER_FILE)
n, p = X.shape
y = np.load(INPUT_Y_CENTER_FILE)

l1_ratios = np.array(proj_predict_MMSE_config.ENET_L1_RATIO_RANGE)
alphas = np.load(INPUT_ALL_GLOBAL_PENALIZATION)

######################
# Recreate CV object #
######################

# Create the cross-validation object
# À DÉFONCER avc proj_predict_MMSE_config.BalancedCV
CV = sklearn.cross_validation.KFold(
    n,
    shuffle=True,
    n_folds=proj_predict_MMSE_config.N_FOLDS,
    random_state=proj_predict_MMSE_config.CV_SEED)

##############################
## Reload results & evaluate #
##############################

print "Evaluating models"
# Reconstruct the global y_true (and compute v)
y_true = []
for fold_index, (train_indices, test_indices) in enumerate(CV):
    #print fold_index
    y_true = np.append(y_true, y[test_indices])

# Compute r_squared for each fold and parameter
r_squared = np.zeros((len(l1_ratios), len(alphas)))
for l1_index, l1_ratio in enumerate(l1_ratios):
    #print l1_ratio
    for alpha_index, alpha in enumerate(alphas):
        #print "\t", alpha
        y_pred = []
        for fold_index, fold_indices in enumerate(CV):
            #print "\t\t", fold_index
            # Reconstruct path
            fold_path = os.path.join(OUTPUT_PATH,
                             proj_predict_MMSE_config.FOLD_PATH_FORMAT.format(
                                 fold_index=fold_index))
            model_path = os.path.join(
                fold_path,
                proj_predict_MMSE_config.ENET_MODEL_PATH_FORMAT.format(
                    l1_ratio=l1_ratio,
                    alpha=alpha))
            y_pred_path = os.path.join(model_path, "y_pred.npy")
            y_pred = np.append(y_pred, np.load(y_pred_path))
        # Compute r_squared
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        r_squared[l1_index, alpha_index] = r2
np.save(OUTPUT_RSQUARED, r_squared)

# Find the optimal parameters
max_index = r_squared.argmax()
max_pos = np.unravel_index(max_index, r_squared.shape)
print l1_ratios[max_pos[0]], alphas[max_pos[1]]
np.save(OUTPUT_OPT_L1_RATIO, l1_ratios[max_pos[0]])
np.save(OUTPUT_OPT_GLOBAL_PENALIZATION, alphas[max_pos[1]])
