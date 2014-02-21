# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:29:56 2014

@author: md238665

Find good regularization parameters for ElasticNet.
They will be used for TV penalized regression.

We don't use ElasticNetCV because it optimizes on MSE and we want to optimize on R^2.

"""

import os

import pickle

import numpy as np
import pandas as pd

import sklearn.cross_validation
import sklearn.linear_model
import sklearn.linear_model.coordinate_descent

import proj_predict_MMSE_config

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"

INPUT_PATH = BASE_PATH
INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")

OUTPUT_PATH = os.path.join(BASE_PATH, "CV")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

OUTPUT_ENET_PATH = os.path.join(OUTPUT_PATH, "ElasticNet")
if not os.path.exists(OUTPUT_ENET_PATH):
    os.makedirs(OUTPUT_ENET_PATH)
OUTPUT_RSQUARED = os.path.join(OUTPUT_ENET_PATH, "r_squared.npy")
OUTPUT_L1_RATIO = os.path.join(OUTPUT_ENET_PATH, "l1_ratio.npy")
# Will be used for TV penalized
OUTPUT_GLOBAL_PENALIZATION = os.path.join(OUTPUT_PATH, "alpha.npy")

#############
# Load data #
#############

X = np.load(INPUT_X_CENTER_FILE)
n, p = X.shape
y = np.load(INPUT_Y_CENTER_FILE)

#########################
# Cross validation loop #
#########################

# Create the cross-validation object
CV = sklearn.cross_validation.KFold(
    n,
    shuffle=True,
    n_folds=proj_predict_MMSE_config.N_FOLDS,
    random_state=proj_predict_MMSE_config.CV_SEED)

# Create l1 ratio grid
l1_ratios = np.array(proj_predict_MMSE_config.ENET_L1_RATIO_RANGE)

# Create alpha grid
mean_l1_ratio = l1_ratios.mean()
alphas = sklearn.linear_model.coordinate_descent._alpha_grid(
    X, y,
    l1_ratio=mean_l1_ratio,
    n_alphas=proj_predict_MMSE_config.N_GLOBAL_PENALIZATION)

# CV
for fold_index, fold_indices in enumerate(CV):
    print fold_index
    fold_path = os.path.join(OUTPUT_PATH,
                             proj_predict_MMSE_config.FOLD_PATH_FORMAT.format(
                                 fold_index=fold_index))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    # Create train and test sets
    train_indices, test_indices = fold_indices
    X_train = X[train_indices]
    np.save(os.path.join(fold_path, "X_train.npy"), X_train)
    X_test = X[test_indices]
    np.save(os.path.join(fold_path, "X_test.npy"), X_test)
    y_train = y[train_indices]
    np.save(os.path.join(fold_path, "y_train.npy"), y_train)
    y_test = y[test_indices]
    np.save(os.path.join(fold_path, "y_test.npy"), y_test)
    for l1_index, l1_ratio in enumerate(l1_ratios):
        print "\t", l1_ratio
        for alpha_index, alpha in enumerate(alphas):
            print "\t\t", alpha
            model_path = os.path.join(
                fold_path,
                proj_predict_MMSE_config.ENET_MODEL_PATH_FORMAT.format(
                    l1_ratio=l1_ratio,
                    alpha=alpha))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # Create and fit model
            model = sklearn.linear_model.ElasticNet(alpha=alpha,
                                                    l1_ratio=l1_ratio,
                                                    fit_intercept=False)
            model.fit(X_train, y_train)
            # Store model
            model_file = os.path.join(model_path, "model.pkl")
            with open(model_file, "w") as f:
                pickle.dump(model, f)
            # Evaluate model (we just return the perdiction and compute a
            #  global r_squared outside the loop in case of unbalanced folds)
            y_pred = model.predict(X_test)
            y_pred_path = os.path.join(model_path,
                                      "y_pred.npy")
            np.save(y_pred_path, y_pred)

#############################
# Reload results & evaluate #
#############################
# TODO: this could be put in another script

print "Evaluating models"
# Reconstruct the global y_true (and compute v)
y_true = []
for fold_index, fold_indices in enumerate(CV):
    #print fold_index
    # Reconstruct path
    fold_path = os.path.join(OUTPUT_PATH,
                     proj_predict_MMSE_config.FOLD_PATH_FORMAT.format(
                         fold_index=fold_index))
    y_true_path = os.path.join(fold_path, "y_test.npy")
    y_true = np.append(y_true, np.load(y_true_path))
v = ((y_true - y_true.mean())**2).mean()

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
        u = ((y_pred - y_true)**2).mean()
        r2 = 1 - u/v
        r_squared[l1_index, alpha_index] = r2
np.save(OUTPUT_RSQUARED, r_squared)

# Find the optimal parameters
max_index = r_squared.argmax()
max_pos = np.unravel_index(max_index, r_squared.shape)
print l1_ratios[max_pos[0]], alphas[max_pos[1]]
np.save(OUTPUT_L1_RATIO, l1_ratios[max_pos[0]])
np.save(OUTPUT_GLOBAL_PENALIZATION, alphas[max_pos[1]])
