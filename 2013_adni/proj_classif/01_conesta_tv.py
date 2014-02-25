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
from parsimony.estimators import RidgeLogisticRegression_L1_TV
import parsimony.algorithms.explicit as algorithms
import time
from sklearn.metrics import precision_recall_fscore_support
from parsimony.datasets import make_classification_struct

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif"
ratio_k, ratio_l, ratio_g = .1, .4, .5

REDUCE = False

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif")
sys.path.append(SRC_PATH)
import utils_proj_classif
#
#
#INPUT_PATH = BASE_PATH
#INPUT_X_CENTER_FILE = os.path.join(INPUT_PATH, "X.center.npy")
#INPUT_Y_CENTER_FILE = os.path.join(INPUT_PATH, "y.center.npy")
#
#INPUT_MASK_PATH = "/neurospin/brainomics/2013_adni/proj_predict_MMSE/SPM/template_FinalQC_MCIc-AD/mask.nii"
#
OUTPUT_PATH = os.path.join(BASE_PATH, "tv")
#if not os.path.exists(OUTPUT_PATH):
#    os.makedirs(OUTPUT_PATH)
#
##############
## Load data #
##############
#
#X = np.load(INPUT_X_CENTER_FILE)
#n, p = X.shape
#y = np.load(INPUT_Y_CENTER_FILE)
#y = y[:, np.newaxis]
#
#mask_im = nibabel.load(INPUT_MASK_PATH)
#mask = mask_im.get_data() != 0
#A, n_compacts = tv.A_from_mask(mask)
mask_im = None

##############
## Load data #
##############
n_samples = 500
shape = (500, 500, 1)
X3d, y, beta3d, proba = make_classification_struct(n_samples=n_samples,
        shape=shape, snr=5, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
A, n_compacts = tv.A_from_shape(beta3d.shape)
#plt.plot(proba[y.ravel() == 1], "ro", proba[y.ravel() == 0], "bo")
#plt.show()

n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]


#ALPHAS = [100, 10, 1, .1]
ALPHAS = [1000, 100, 50, 10, 5, 1, 0.1]


###############################
# Iterate over Hyper-parameters
###############################

def mapper(Xtr, ytr, Xte, yte, A, alpha, ratio_k, ratio_l, ratio_g, mask_im):
    #alpha = 10
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, output=True,
                               algorithm=algorithms.StaticCONESTA(max_iter=500))
    tv.fit(Xtr, ytr, beta)
    y_pred_tv = tv.predict(Xte)
    #y_pred = yte
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(mod.info["t"]), np.sum(mod.info["t"]))
    out_dir = os.path.join(OUTPUT_PATH,
                 "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print out_dir, "Time ellapsed:", time.time() - time_curr
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
    time_curr = time.time()
    utils_proj_classif.save_model(out_dir, tv, mask_im)
    np.save(os.path.join(out_dir, "y_pred_tv.npy"), y_pred_tv)
    np.save(os.path.join(out_dir, "y_true.npy"), yte)


Parallel(n_jobs=len(ALPHAS), verbose=True)(
    delayed(mapper) (Xtr, ytr, Xte, yte, A, alpha, ratio_k, ratio_l, ratio_g, mask_im)
    for alpha in ALPHAS)

#########################
# Result: reduce
#########################

if REDUCE:
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

#    
#    In [108]: Out[108]: 
#    {0.1: 31.36674448088975,
#     1: 27.745439565579151,
#     10: 28.343674174636991,
#     100: 28.343675827156019}
#    
#    In [109]: Out[109]: 
#    {0.1: -0.10665760757931086,
#     1: 0.021106516502308326,
#     10: 5.8303100791690099e-08,
#     100: 1.8252066524837574e-13}
#    
#    plt.plot(y[.1]['y_true'], y[.1]['y_pred'], "bo") 
#    plt.plot(y[1]['y_true'], y[1]['y_pred'], "bo") 
#    plt.plot(y[10]['y_true'], y[10]['y_pred'], "bo") 
#    plt.plot(y[100]['y_true'], y[100]['y_pred'], "bo") 
#    plt.show()
#    
#    np.corrcoef(y[100]['y_true'], y[100]['y_pred'])
