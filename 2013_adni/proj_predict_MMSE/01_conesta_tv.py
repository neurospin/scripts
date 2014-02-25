# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""


import os, glob
import sys
#import pickle
import numpy as np
#import pandas as pd
from joblib import Parallel, delayed
import pylab as plt
#import sklearn.cross_validation
#import sklearn.linear_model
#import sklearn.linear_model.coordinate_descent
import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import RidgeRegression_L1_TV
import parsimony.algorithms.explicit as algorithms
import nibabel
import time
from sklearn.metrics import r2_score, mean_squared_error

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE"
alpha = 10.
ratio_k, ratio_l, ratio_g = .1, .4, .5


SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_predict_MMSE")
sys.path.append(SRC_PATH)
import proj_predict_MMSE_config


INPUT_PATH = BASE_PATH
INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")

INPUT_MASK_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE/SPM/template_FinalQC_MCIc-AD/mask.nii"

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

#ALPHAS = [100, 10, 1, .1]
ALPHAS = [1000, 100, 10, 1]

#########################
# Fit on all data       #
#########################

if False:
    alpha = 100
    alpha = 1000    
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeRegression_L1_TV(k, l, g, A, 
                                      algorithm=algorithms.StaticCONESTA(max_iter=500))
    tv.fit(X, y)
    #CPU times: user 43687.42 s, sys: 4.67 s, total: 43692.09 s
    #Wall time: 43672.06 s
    out_dir = os.path.join(OUTPUT_PATH, "all",
                     "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    proj_predict_MMSE_config.save_model(out_dir, tv, mask_im)
    y_pred = tv.predict(X)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(out_dir, "y_true.npy"), y)
    r2_score(y, y_pred)
    plt.plot(y, y_pred, "bo")
    plt.show()

#########################
# Cross validation loop #
#########################

#alphas = [1]


def mapper(X, y, fold, train, test, A, alphas, ratio_k, ratio_l, ratio_g, mask_im):
    print "** FOLD **", fold
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
                                   algorithm=algorithms.StaticCONESTA(max_iter=500))
        tv.fit(Xtr, ytr, beta)
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

Parallel(n_jobs=5, verbose=True)(
    delayed(mapper) (X, y, fold, train, test,A, alphas, ratio_k, ratio_l, ratio_g, mask_im)
    for fold, (train, test) in enumerate(CV))

#########################
# Result: reduce
#########################

alphas = [100, 10, 1, .1]

y = dict()
r2_tot = dict()
mse_tot = dict()
r2_mean = dict()

for alpha in alphas:
    y_pred  = list()
    y_true = list()
    r2 = list()
    for rep in glob.glob(os.path.join(OUTPUT_PATH, "cv", "*", str(alpha)+"*")):
        print rep
        #rep = '/neurospin/brainomics/2013_adni/proj_predict_MMSE/tv/cv/2/100-0.1-0.4-0.5'
        y_pred_f = np.load(os.path.join(rep, "y_pred.npy"))
        y_true_f = np.load(os.path.join(rep, "y_true.npy"))
        r2.append(r2_score(y_true_f, y_pred_f))
        y_pred.append(y_pred_f.ravel())
        y_true.append(y_true_f.ravel())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    y[alpha] = dict(y_true=y_true, y_pred=y_pred)
    r2_tot[alpha] = r2_score(y_true, y_pred)
    r2_mean[alpha] = np.mean(r2)
    mse_tot[alpha] = mean_squared_error(y_true, y_pred)

mse_tot
r2_tot

In [108]: Out[108]: 
{0.1: 31.36674448088975,
 1: 27.745439565579151,
 10: 28.343674174636991,
 100: 28.343675827156019}

In [109]: Out[109]: 
{0.1: -0.10665760757931086,
 1: 0.021106516502308326,
 10: 5.8303100791690099e-08,
 100: 1.8252066524837574e-13}

plt.plot(y[.1]['y_true'], y[.1]['y_pred'], "bo") 
plt.plot(y[1]['y_true'], y[1]['y_pred'], "bo") 
plt.plot(y[10]['y_true'], y[10]['y_pred'], "bo") 
plt.plot(y[100]['y_true'], y[100]['y_pred'], "bo") 
plt.show()

np.corrcoef(y[100]['y_true'], y[100]['y_pred'])

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