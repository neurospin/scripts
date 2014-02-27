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
import nibabel

#import sklearn.cross_validation
#import sklearn.linear_model
#import sklearn.linear_model.coordinate_descent
import parsimony.functions.nesterov.tv as tv
from parsimony.estimators import RidgeLogisticRegression_L1_TV
import parsimony.algorithms.explicit as algorithms
import time
from sklearn.metrics import precision_recall_fscore_support
from parsimony.datasets import make_classification_struct
from parsimony.utils import plot_map2d

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
SIMU = False

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
if SIMU:
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


##############
## Load data #
##############
INPUT_X_TRAIN_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.train.center.npy")
INPUT_X_TEST_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.test.center.npy")
INPUT_Y_TRAIN_FILE = os.path.join(BASE_PATH, "y_CTL_AD.train.npy")
INPUT_Y_TEST_FILE = os.path.join(BASE_PATH, "y_CTL_AD.test.npy")
INPUT_MASK_PATH = os.path.join(BASE_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                          "mask.nii")
mask_im = nibabel.load(INPUT_MASK)
mask = mask_im.get_data() != 0
A, n_compacts = tv.A_from_mask(mask)

Xtr = np.load(INPUT_X_TRAIN_CENTER_FILE)
Xte = np.load(INPUT_X_TEST_CENTER_FILE)
ytr = np.load(INPUT_Y_TRAIN_FILE)[:, np.newaxis]
yte = np.load(INPUT_Y_TEST_FILE)[:, np.newaxis]

#ALPHAS = [100, 10, 1, .1]
ALPHAS = [100, 50, 10, 5, 1, 0.1]
#ALPHAS = [1]


###############################
# Iterate over Hyper-parameters
###############################

def mapper(alpha, ratio_k, ratio_l, ratio_g, mask_im):
    #alpha = 10
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, output=True,
                               algorithm=algorithms.StaticCONESTA(max_iter=500))
    tv.fit(Xtr, ytr, beta)
    y_pred_tv = tv.predict(Xte)
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    out_dir = os.path.join(OUTPUT_PATH,
                 "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print out_dir, "Time ellapsed:", time.time() - time_curr, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
    time_curr = time.time()
    utils_proj_classif.save_model(out_dir, tv, beta, mask_im,
                                  y_pred_tv=y_pred_tv,
                                  y_true=yte)
    #np.save(os.path.join(out_dir, "y_pred_tv.npy"), y_pred_tv)
    #np.save(os.path.join(out_dir, "y_true.npy"), yte)


Parallel(n_jobs=len(ALPHAS), verbose=True)(
    delayed(mapper) (alpha, ratio_k, ratio_l, ratio_g, mask_im)
    for alpha in ALPHAS)

#########################
# Result: reduce
#########################

if REDUCE:
    y = dict()
    recall_tot = dict()
    models = dict()
    #mse_tot = dict()
    #r2_mean = dict()
    for rep in glob.glob(os.path.join(OUTPUT_PATH, "*-*-*")):
        key = os.path.basename(rep)
        print rep
        res = utils_proj_classif.load(rep)
        #rep = '/neurospin/brainomics/2013_adni/proj_predict_MMSE/tv/cv/2/100-0.1-0.4-0.5'
        y_pred = res["y_pred_tv"].ravel()
        y_true = res["y_true"].ravel()
        y[key] = dict(y_true=y_true, y_pred=y_pred)
        _, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        recall_tot[key] = r
        models[key] = res['model']

    print recall_tot
    
    key = '10-0.1-0.4-0.5'
    tv = models[key]
    title = key+", ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    print tv.beta.min(), tv.beta.max()
    plot_map2d(beta3d.squeeze(), title="betastar", limits=[beta3d.min(), beta3d.max()])
    plot_map2d(tv.beta.reshape(shape), title=title, limits=[beta3d.min(), beta3d.max()])
    plt.show()
    