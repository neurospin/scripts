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
OUTPUT_PATH = os.path.join(BASE_PATH, "tv")
INPUT_X_TRAIN_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.train.center.npy")
INPUT_X_TEST_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.test.center.npy")
INPUT_Y_TRAIN_FILE = os.path.join(BASE_PATH, "y_CTL_AD.train.npy")
INPUT_Y_TEST_FILE = os.path.join(BASE_PATH, "y_CTL_AD.test.npy")
INPUT_MASK_PATH = os.path.join(BASE_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                              "mask.nii")

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif")
sys.path.append(SRC_PATH)
import utils_proj_classif


MODE = "split"



## ARGS ===================================================================
#ratio_k, ratio_l, ratio_g = .1, .1, .8
#ALPHAS = [100, 10, 1, .1]
#ALPHAS = [5, 1, 0.5, 0.1]
#ALPHAS = " ".join([str(a) for a in ALPHAS])
ALPHAS = "5 1 0.5 0.1"
RATIOS_LIST = "0.1 0.1 0.8; 0.0 0.2 0.8; 0.2 0.0 0.8"
NBCORES = 8

parser = optparse.OptionParser(description=__doc__)
parser.add_option('--mode',
    help='Execution mode: "simu" (simulation data), "split" (train,test), \
    "all" (train=train+test), "reduce" (default %s)' % MODE, default=MODE, type=str)
parser.add_option('--alphas',
    help='alphas values (default %s)' % ALPHAS, default=ALPHAS, type=str)
parser.add_option('--cores',
    help='Nb cpu cores to use (default %i)' % NBCORES, default=NBCORES, type=int)
parser.add_option('--ratios',
    help='Ratio triplet separated by ";" (default %s)' % RATIOS_LIST, default=RATIOS_LIST, type=str)


options, args = parser.parse_args(sys.argv)

MODE = options.mode
ALPHAS  = options.alphas
RATIOS_LIST = options.ratios
ALPHAS = [float(a) for a in ALPHAS.split()]
RATIOS_LIST = [[float(ratio) for ratio in ratios.split()]  for ratios in RATIOS_LIST.split(";")]
PARAMS_LIST = [[alpha]+ratios for ratios in RATIOS_LIST for alpha in ALPHAS]
NBCORES = options.cores

#print MODE, PARAMS_LIST, NBCORES

##############
## Load data
##############
if MODE == "simu":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "simu")
    n_samples = 500
    shape = (10, 10, 1)
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
    mask_im = nibabel.load(INPUT_MASK)
    mask = mask_im.get_data() != 0
    A, n_compacts = tv.A_from_mask(mask)
    
if MODE == "split":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "split")

if MODE == "all":
    OUTPUT_PATH = os.path.join(OUTPUT_PATH, "all")
    Xtr = Xte = np.r_[Xtr, Xte]
    ytr = yte = np.r_[ytr, yte]

weigths = np.zeros(ytr.shape)
prop = np.asarray([np.sum(ytr == l) for l in [0, 1]]) / float(ytr.size)
weigths[ytr==0] = prop[1]
weigths[ytr==1] = prop[0]

print "weitghed sum", weigths[ytr==0].sum(), weigths[ytr==1].sum()

###############################
# Mapper
###############################

def mapper(params):
    alpha, ratio_k, ratio_l, ratio_g = params
    out_dir = os.path.join(OUTPUT_PATH,
                 "-".join([str(v) for v in (alpha, ratio_k, ratio_l, ratio_g)]))
    print "START:", out_dir
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
    tv.function = tv.A = None # Save disk space
    utils_proj_classif.save_model(out_dir, tv, beta, mask_im,
                                  y_pred_tv=y_pred_tv,
                                  y_true=yte)

###############################
# Execution
###############################

print PARAMS_LIST
p = Pool(NBCORES)
p.map(mapper, PARAMS_LIST)


#########################
# Result: reduce
#########################

if MODE == "reduce":
    #print MODE, ALPHAS, ratio_k, ratio_l, ratio_g
    OUTPUT_PATH = os.path.join(BASE_PATH, "tv", MODE)
    y = dict()
    recall_tot = dict()
    models = dict()
    #mse_tot = dict()
    #r2_mean = dict()
    for rep in glob.glob(os.path.join(OUTPUT_PATH, "*-*-*")):
        key = os.path.basename(rep)
        print rep
        res = utils_proj_classif.load(rep)
        mod = res['model']
        #self.function = None
        #self.A = None
        #import pickle
        #pickle.dump(self, open(os.path.join(rep, "model.pkl"), "w"))
        #rep = '/neurospin/brainomics/2013_adni/proj_predict_MMSE/tv/cv/2/100-0.1-0.4-0.5'
        # BUG CORRECT START
        arr = np.zeros(mask.shape)
        arr[mask] = mod.beta.ravel()
        im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine())#, header=mask_im.get_header().copy())
        im_out.to_filename(os.path.join(rep,"beta.nii"))
        print np.all(nibabel.load(os.path.join(rep,"beta.nii")).get_data()[mask] == mod.beta.ravel())
        # BUG CORRECT END
        y_pred = res["y_pred_tv"].ravel()
        y_true = res["y_true"].ravel()
        y[key] = dict(y_true=y_true, y_pred=y_pred)
        _, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        recall_tot[key] = r
        models[key] = res['model']
    r =list()
    for k in recall_tot: r.append(k.split("-")+recall_tot[k].tolist())
    import pandas as pd
    res = pd.DataFrame(r, columns=["alpha", "l2_ratio", "l1_ratio", "tv_ratio",  "recall_0", "recall_1"])
    res = res.sort("recall_1", ascending=False)
    print res.to_string()
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

# TV 0.9 => 0.6
#python 02_conesta_tv.py --cores=8 --ratios="0.05 0.05 0.9; 0.0 0.1 0.9; 0.1 0.0 0.9; 0.1 0.1 0.8; 0.0 0.2 0.8; 0.2 0.0 0.8; 0.2 0.2 0.6; 0.4 0.0 0.6; 0.0 0.4 0.6"
# TV 0.95 et 0.5
#python 02_conesta_tv.py --cores=8 --ratios="0.025 0.025 0.9; 0.0 0.05 0.95; 0.05 0.0 0.9; 0.25 0.25 0.5; 0.5 0.0 0.5; 0.0 0.5 0.5

# Execution time
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/50-0.1-0.1-0.8 Time ellapsed: 1357.92806792 ite:1289, time:479.500000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/10-0.1-0.1-0.8 Time ellapsed: 6142.82701111 ite:11019, time:3397.590000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/5-0.1-0.1-0.8 Time ellapsed: 6218.87577105 ite:11401, time:3501.290000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/1-0.1-0.1-0.8 Time ellapsed: 6580.30336094 ite:12270, time:3830.900000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/0.5-0.1-0.1-0.8 Time ellapsed: 6619.85201001 ite:12412, time:3854.080000
#/neurospin/brainomics/2013_adni/proj_classif/tv/split/0.1-0.1-0.1-0.8 Time ellapsed: 6710.06726408 ite:12703, time:3973.000000
