# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""


import os
import sys
#import pickle
import numpy as np
#import pandas as pd
from joblib import Parallel, delayed
#import sklearn.cross_validation
#import sklearn.linear_model
#import sklearn.linear_model.coordinate_descent
import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import RidgeRegression_L1_TV
import parsimony.algorithms.explicit as algorithms
import nibabel
import time
from sklearn.metrics import r2_score

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"
alpha = 10.
ratio_k, ratio_l, ratio_g = .1, .4, .5


SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_predict_MMSE")
sys.path.append(SRC_PATH)
import proj_predict_MMSE_config


INPUT_PATH = BASE_PATH
INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")

INPUT_MASK_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE/SPM/template_FinalQC_MCIc-AD/mask.img"

OUTPUT_PATH = os.path.join(BASE_PATH, "tv")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#############
# Load data #
#############

X = np.load(INPUT_X_CENTER_FILE)
n, p = X.shape
y = np.load(INPUT_Y_CENTER_FILE)
y = y[:, np.newaxis]

mask_im = nibabel.load(INPUT_MASK_PATH)
mask = mask_im.get_data() != 0
A, n_compacts = tv.A_from_mask(mask)



#########################
# Fit on all data       #
#########################

if False:
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    
    tv = RidgeRegression_L1_TV(k, l, g, A, 
                                      algorithm=algorithms.StaticCONESTA(max_iter=1000))
    beta = tv.start_vector.get_vector((X.shape[1], 1))
    %time tv.fit(X, y)
    #CPU times: user 43687.42 s, sys: 4.67 s, total: 43692.09 s
    #Wall time: 43672.06 s
    out = os.path.join(OUTPUT_PATH, "all",
                     "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    
    proj_predict_MMSE_config.save_model(out, tv, mask_im)

#########################
# Cross validation loop #
#########################

alphas = [1000, 100, 10, 1, .1]
#alphas = [1]


def mapper(X, y, fold, train, test, A, alphas, ratio_k, ratio_l, ratio_g, mask_im):
    #print "** FOLD **", fold
    Xtr = X[train, :]
    Xte = X[test, :]
    ytr = y[train, :]
    yte = y[test, :]
    n_train = Xtr.shape[0]
    time_curr = time.time()
    beta = None
    for alpha in alphas:
        k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
        tv = RidgeRegression_L1_TV(k, l, g, A, output=True,
                                   algorithm=algorithms.StaticCONESTA(max_iter=1000))
        tv.fit(X, y, beta)
        y_pred = tv.predict(Xte)
        #y_pred = yte
        beta = tv.beta
        #print key, "ite:%i, time:%f" % (len(mod.info["t"]), np.sum(mod.info["t"]))
        out_dir = os.path.join(OUTPUT_PATH, "cv", str(fold),
                     "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
        print out_dir, "Time ellapsed:", time.time() - time_curr
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        time_curr = time.time()
        proj_predict_MMSE_config.save_model(out_dir, tv, mask_im)
        np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
        np.save(os.path.join(out_dir, "y_true.npy"), yte)


CV = proj_predict_MMSE_config.BalancedCV(y, proj_predict_MMSE_config.N_FOLDS,
    random_seed=proj_predict_MMSE_config.CV_SEED)

#for fold, (train, test) in enumerate(CV):
#    print fold
#mapper(X, y, fold, train, test, A, alphas, ratio_k, ratio_l, ratio_g, mask_im)

Parallel(n_jobs=proj_predict_MMSE_config.N_FOLDS, verbose=True)(
    delayed(mapper) (X, y, fold, train, test,A, alphas, ratio_k, ratio_l, ratio_g, mask_im)
    for fold, (train, test) in enumerate(CV))




# First RUN no limit on iteration
#Time: 37018.9876552 # 10h
#Time: 18923.3399591 # 5h
#Time: 18117.568048
#Time: 19425.1853101
#Time: 19714.5431721
# warm retart accross folds divide by 2 execution time
# 

#y_pred = np.concatenate(y_pred, axis=0)
#y_true = np.concatenate(y_true, axis=0)
#
#r2_score(y_true, y_pred)
##-0.0823733933819768
#
#for i, (train, test) in enumerate(CV):
#    Xtr = X[train, :]
#    Xte = X[test, :]
#    ytr = y[train, :]
#    yte = y[test, :]
#    print ytr.mean(), yte.mean(), np.concatenate([ytr, yte]).mean()