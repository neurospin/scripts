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

import sklearn.cross_validation
import sklearn.linear_model
import sklearn.linear_model.coordinate_descent

from joblib import Parallel, delayed

import proj_predict_MMSE_config

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"

INPUT_PATH = BASE_PATH
INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")

OUTPUT_PATH = os.path.join(BASE_PATH, "ElasticNet")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_ALL_GLOBAL_PENALIZATION = os.path.join(OUTPUT_PATH, "alphas.npy")

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
# À DÉFONCER avc proj_predict_MMSE_config.BalancedCV
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
np.save(OUTPUT_ALL_GLOBAL_PENALIZATION, alphas)


def fit_and_predict(model, model_file, X_train, y_train, X_test, y_pred_path):
    model.fit(X_train, y_train)
    # Store
    with open(model_file, "w") as f:
        pickle.dump(model, f)
    # Evaluate model (we just return the prediction and compute a
    #  global r_squared outside the loop in case of unbalanced folds)
    y_pred = model.predict(X_test)
    np.save(y_pred_path, y_pred)

# Create CV jobs
print "Creating jobs"
MODELS = []
MODEL_FILES = []
X_TRAINS = []
Y_TRAINS = []
X_TESTS = []
PRED_FILES = []
for fold_index, fold_indices in enumerate(CV):
    #print fold_index
    fold_path = os.path.join(OUTPUT_PATH,
                             proj_predict_MMSE_config.FOLD_PATH_FORMAT.format(
                                 fold_index=fold_index))
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
    # Create train and test sets
    train_indices, test_indices = fold_indices
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    for l1_index, l1_ratio in enumerate(l1_ratios):
        #print "\t", l1_ratio
        for alpha_index, alpha in enumerate(alphas):
            #print "\t\t", alpha
            model_path = os.path.join(
                fold_path,
                proj_predict_MMSE_config.ENET_MODEL_PATH_FORMAT.format(
                    l1_ratio=l1_ratio,
                    alpha=alpha))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            # Append model
            model = sklearn.linear_model.ElasticNet(alpha=alpha,
                                                    l1_ratio=l1_ratio,
                                                    fit_intercept=False)
            MODELS.append(model)
            # Append data
            X_TRAINS.append(X_train)
            Y_TRAINS.append(y_train)
            X_TESTS.append(X_test)
            # Append output files
            model_file = os.path.join(model_path, "model.pkl")
            MODEL_FILES.append(model_file)
            y_pred_path = os.path.join(model_path,
                                      "y_pred.npy")
            PRED_FILES.append(y_pred_path)

ALL_JOBS = zip(MODELS,
               MODEL_FILES,
               X_TRAINS,
               Y_TRAINS,
               X_TESTS,
               PRED_FILES)

# Run it
print "Running jobs"
Parallel(n_jobs=4, verbose=True)(
    delayed(fit_and_predict)(*j) for j in ALL_JOBS)
