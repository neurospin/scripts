# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""


import os

import pickle

import numpy as np
import pandas as pd

import sklearn.cross_validation
import sklearn.linear_model
import sklearn.linear_model.coordinate_descent
import sys
import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import RidgeRegression_L1_TV
import nibabel
import time

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_predict_MMSE")
sys.path.append(SRC_PATH)
import proj_predict_MMSE_config


INPUT_PATH = BASE_PATH
INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")

INPUT_MASK_PATH = "/neurospin/brainomics/2013_adni/masks/template_FinalQC_MCIc-AD/mask.img"

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
y = y[:, np.newaxis]

babel_mask = nibabel.load(INPUT_MASK_PATH)
mask = babel_mask.get_data() != 0

alpha_g = 10.
k, l, g = alpha_g * np.array((.1, .4, .5))  # l2, l1, tv penalties
A, n_compacts = tv.A_from_mask(mask)

ridgel1tv = RidgeRegression_L1_TV(k, l, g, A)
beta = ridgel1tv.start_vector.get_vector((X.shape[1], 1))
if False:
    %time ridgel1tv.fit(X, y)


#########################
# Cross validation loop #
#########################

# Create the cross-validation object
CV = sklearn.cross_validation.KFold(
    n,
    shuffle=True,
    #n_folds=2,
    n_folds=proj_predict_MMSE_config.N_FOLDS,
    random_state=proj_predict_MMSE_config.CV_SEED)


predictions = list()
time_curr = time.time()
for train, test in CV:
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    n_train = Xtr.shape[0]
    #print train, test
#    enet = sklearn.linear_model.ElasticNet(alpha=alpha_g / (2. * n_train),
#                                            l1_ratio=.5,
#                                            fit_intercept=False)
#    enet.fit(Xtr, ytr)
#    predictions.append(enet.predict(Xte))
    ridgel1tv.fit(Xtr, ytr, beta=beta)
    predictions.append(ridgel1tv.predict(Xte))
    print "Time:", time.time() - time_curr
    time_curr = time.time()
