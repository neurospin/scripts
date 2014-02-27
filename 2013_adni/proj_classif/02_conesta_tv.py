# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""


import os, glob
import sys, optparse
#import pickle
import numpy as np
#import pandas as pd
from multiprocessing import Pool
#from joblib import Parallel, delayed
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

## GLOBALS ==================================================================
BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif"
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

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif")
sys.path.append(SRC_PATH)
import utils_proj_classif

## ARGS ===================================================================
MODE = "split"
ratio_k, ratio_l, ratio_g = .1, .1, .8
#ALPHAS = [100, 10, 1, .1]
ALPHAS = [100, 50, 10, 5, 1, 0.5, 0.1]
ALPHAS = " ".join([str(a) for a in ALPHAS])

parser = optparse.OptionParser(description=__doc__)
parser.add_option('--mode',
    help='Execution mode: "simu" (simulation data), "split" (train,test), \
    "all" (train=train+test), "reduce" (default %s)' % MODE, default=MODE, type=str)
parser.add_option('--alphas',
    help='alphas values (default %s)' % ALPHAS, default=ALPHAS, type=str)
parser.add_option('--ratio_k',
    help='l2 penalty ratio (default %f)' % ratio_k, default=ratio_k, type=float)
parser.add_option('--ratio_l',
    help='l1 penalty ratio (default %f)' % ratio_l, default=ratio_l, type=float)
parser.add_option('--ratio_g',
    help='tv penalty ratio (default %f)' % ratio_g, default=ratio_g, type=float)

options, args = parser.parse_args(sys.argv)

MODE = options.mode
ALPHAS  = options.alphas
ALPHAS = [float(a) for a in ALPHAS.split()]
ratio_k = options.ratio_k
ratio_l = options.ratio_l
ratio_g = options.ratio_g

#print MODE, ALPHAS, ratio_k, ratio_l, ratio_g
OUTPUT_PATH = os.path.join(BASE_PATH, "tv")



##############
## Load data #
##############
if MODE == "simu":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "simu")
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
    mask_im = None
else:
    Xtr = np.load(INPUT_X_TRAIN_CENTER_FILE)
    Xte = np.load(INPUT_X_TEST_CENTER_FILE)
    ytr = np.load(INPUT_Y_TRAIN_FILE)[:, np.newaxis]
    yte = np.load(INPUT_Y_TEST_FILE)[:, np.newaxis]
    A, n_compacts = tv.A_from_mask(mask)
    
if MODE == "split":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "split")

if MODE == "all":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "all")
    Xtr = Xte = np.r_[Xtr, Xte]
    ytr = yte = np.r_[ytr, yte]

weigths = np.zeros(ytr.shape)
prop = np.asarray([np.sum(ytr == l) for l in [0, 1]]) / float(ytr.size)
weigths[ytr==0] = prop[0]
weigths[ytr==1] = prop[1]


###############################
# Iterate over Hyper-parameters
###############################

def mapper(alpha):
    out_dir = os.path.join(OUTPUT_PATH,
                 "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print "START:", out_dir
    #alpha = 10
    np.asarray([np.sum(ytr == l) for l in np.unique(ytr)]) / float(ytr.size)
    time_curr = time.time()
    beta = None
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g)) # l2, l1, tv penalties
    tv = RidgeLogisticRegression_L1_TV(k, l, g, A, weigths=weigths, output=True,
                               algorithm=algorithms.StaticCONESTA(max_iter=500))
    tv.fit(Xtr, ytr)#, beta)
    y_pred_tv = tv.predict(Xte)
    beta = tv.beta
    #print key, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    print out_dir, "Time ellapsed:", time.time() - time_curr, "ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
    time_curr = time.time()
    utils_proj_classif.save_model(out_dir, tv, beta, mask_im,
                                  y_pred_tv=y_pred_tv,
                                  y_true=yte)


p = Pool(len(ALPHAS))

p.map(mapper, ALPHAS)
#Parallel(n_jobs=len(ALPHAS), verbose=True)(
#    delayed(mapper) (alpha, ratio_k, ratio_l, ratio_g)
#    for alpha in ALPHAS)

#########################
# Result: reduce
#########################

if MODE == "reduce":
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
    for k in recall_tot: print k, "\t", recall_tot[k]
    #[m.weigths[:2] for m in models.values()]
    #key = '1-0.1-0.4-0.5'
    key = '0.1-0.1-0.4-0.5'
    #key = '5-0.1-0.4-0.5'
    recall_tot[key]
    np.all(y[key]["y_true"] == yte.ravel())
    tv = models[key]
    precision_recall_fscore_support(yte, tv.predict(Xte))
    precision_recall_fscore_support(ytr, tv.predict(Xtr))
    
    if SIMU:
        key = '10-0.1-0.4-0.5'
        tv = models[key]
        title = key+", ite:%i, time:%f" % (len(tv.info["t"]), np.sum(tv.info["t"]))
        print tv.beta.min(), tv.beta.max()
        plot_map2d(beta3d.squeeze(), title="betastar", limits=[beta3d.min(), beta3d.max()])
        plot_map2d(tv.beta.reshape(shape), title=title, limits=[beta3d.min(), beta3d.max()])
        plt.show()

# python 02_conesta_tv.py --alphas="10 5 1 0.5 0.1 0.05" --ratio_k=0.25 --ratio_l=0.1 --ratio_g=0.65



#0.1-0.1-0.4-0.5 	[ 0.78947368  0.58064516]
#50-0.1-0.1-0.8 	[ 0.31578947  0.75806452]
#100-0.1-0.4-0.5 	[ 0.  1.]
#10-0.1-0.4-0.5 	[ 0.71052632  0.41935484]
#5-0.1-0.4-0.5 	[ 0.5         0.58064516]
#50-0.1-0.4-0.5 	[ 0.  1.]
#1-0.1-0.4-0.5 	[ 0.76315789  0.53225806]
#0.5-0.1-0.4-0.5 	[ 0.78947368  0.58064516]
#10-0.1-0.1-0.8 	[ 0.46052632  0.61290323]
#5-0.1-0.1-0.8 	[ 0.63157895  0.41935484]
#100-0.1-0.1-0.8 	[ 0.  1.]


#/neurospin/brainomics/2013_adni/proj_classif/tv/split/50-0.1-0.1-0.8 Time ellapsed: 1357.92806792 ite:1289, time:479.500000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/10-0.1-0.1-0.8 Time ellapsed: 6142.82701111 ite:11019, time:3397.590000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/5-0.1-0.1-0.8 Time ellapsed: 6218.87577105 ite:11401, time:3501.290000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/1-0.1-0.1-0.8 Time ellapsed: 6580.30336094 ite:12270, time:3830.900000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/0.5-0.1-0.1-0.8 Time ellapsed: 6619.85201001 ite:12412, time:3854.080000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/0.1-0.1-0.1-0.8 Time ellapsed: 6710.06726408 ite:12703, time:3973.000000
