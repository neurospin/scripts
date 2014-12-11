# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:39:02 2014

@author: cp243490
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
from parsimony.algorithms.utils import Info
import parsimony.functions.nesterov.tv as tv_helper
import shutil
from scipy import sparse
from scipy.stats import binom_test

from collections import OrderedDict


NFOLDS = 5


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.PROB_CLASS1 = config["prob_class1"]
    GLOBAL.PENALTY_START = config["penalty_start"]
    STRUCTURE = nibabel.load(config["structure"])
    GLOBAL.A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.STRUCTURE = STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...]
                        for idx in resample]
                            for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, Xtest], "y":[ytrain, ytest]}
    # key: list of parameters
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    n_voxels = np.count_nonzero(STRUCTURE.get_data())
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)
    mask = STRUCTURE.get_data() != 0
    Xtr_r = Xtr
    Xte_r = Xte
    A = GLOBAL.A
    info = [Info.num_iter]
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight,
                                   algorithm_params={'info': info})
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    proba_pred = mod.predict_probability(Xte_r)  # a posteriori probability
    beta = mod.beta
    ret = dict(y_pred=y_pred, proba_pred=proba_pred, y_true=yte,
               beta=beta,  mask=mask,
               n_iter=mod.get_info()['num_iter'])
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    import mapreduce as GLOBAL
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    # import glob, mapreduce
    # values = [mapreduce.OutputCollector(p)
    #        for p in glob.glob("/neurospin/brainomics/2014_deptms/MRI/results/*/0.05_0.45_0.45_0.1_-1.0/")]
    # Compute sd; ie.: compute results on each folds
    penalty_start = GLOBAL.PENALTY_START
    prob_class1 = GLOBAL.PROB_CLASS1
    values = [item.load() for item in values[1:]]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
            item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
            / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred)  # area under curve score.
    n_ite = np.mean(np.array([item["n_iter"] for item in values]))
    betas = np.hstack([item["beta"][penalty_start:]  for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    success = r * s
    success = success.astype('int')
    pvalue_class0 = binom_test(success[0], s[0], 1 - prob_class1)
    pvalue_class1 = binom_test(success[1], s[1], prob_class1)
    a, l1, l2 = float(key[0]), float(key[1]), float(key[2])
    tv, k_ratio = float(key[3]), float(key[4])
    left = float(1 - tv)
    if left == 0:
        left = 1.
    scores = OrderedDict()
    scores['a'] = a
    scores['l1'] = l1
    scores['l2'] = l2
    scores['l1l2_ratio'] = l1 / left
    scores['tv'] = tv
    scores['k_ratio'] = k_ratio
    scores['recall_0'] = r[0]
    scores['pvalue_recall_0'] = pvalue_class0
    scores['recall_1'] = r[1]
    scores['pvalue_recall_1'] = pvalue_class1
    scores['max_pvalue_recall'] = np.maximum(pvalue_class0, pvalue_class1)
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['n_ite_mean'] = n_ite
    scores['auc'] = auc
    scores['beta_cor_mean'] = beta_cor_mean
    # proportion of non zeros elements in betas matrix over all folds
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas)) / \
                                    float(np.prod(betas.shape))
    return scores


##############################################################################
## Run all
def run_all(config):
    import mapreduce
    BASE_PATH = "/neurospin/brainomics/2014_deptms"
    OUTPUT_ENETTV = os.path.join(BASE_PATH,   "results_enettv")
    params = config["params"][0]
    key = '_'.join([str(p) for p in params])
    modality = config["modality"]
    roi = config["roi"]
    WD = os.path.join(OUTPUT_ENETTV, modality + '_' + roi)
    #class GLOBAL: DATA = dict()
    load_globals(config)
    OUTPUT = os.path.join(WD, 'test', key)
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
    # run /home/ed203246/bin/mapreduce.py
    oc = mapreduce.OutputCollector(OUTPUT)
    X = np.load(os.path.join(WD,
                             'X_' + modality + '_' + roi + '.npy'))
    y = np.load(os.path.join(WD,  'y.npy'))
    mapreduce.DATA_RESAMPLED = {}
    mapreduce.DATA_RESAMPLED["X"] = [X, X]
    mapreduce.DATA_RESAMPLED["y"] = [y, y]
    params = np.array([float(p) for p in key.split("_")])
    mapper(params, oc)
    #oc.collect(key=key, value=ret)


if __name__ == "__main__":

   #########################################################################
    ## load data
   MASKDEP_PATH = "/neurospin/brainomics/2014_deptms/maskdep"

   DATASET_PATH = os.path.join(MASKDEP_PATH,    "datasets")
   ENETTV_PATH = os.path.join(MASKDEP_PATH, 'results_enettv')
   if not os.path.exists(ENETTV_PATH):
       os.makedirs(ENETTV_PATH)

   penalty_start = 3
   dilatation = ["dilatation_masks", "dilatation_within-brain_masks"]

#########################################################################
## Build config file for all couple (Modality, roi)

for dilat in dilatation:
    if dilat == "dilatation_masks":
        dilat_process = "dilatation"
    elif dilat == "dilatation_within-brain_masks":
        dilat_process = "dilatation_within-brain"
    WD = os.path.join(ENETTV_PATH, dilat)

    if not os.path.exists(WD):
        os.makedirs(WD)

    INPUT_DATA_X = os.path.join(DATASET_PATH, dilat,
                                'X_' + dilat_process + '.npy')
    INPUT_DATA_y = os.path.join(DATASET_PATH, dilat,
                                'y.npy')
    INPUT_MASK = os.path.join(DATASET_PATH, dilat,
                              'mask_' + dilat_process + '.nii')
    # copy X, y, mask file names in the current directory
    shutil.copy2(INPUT_DATA_X, os.path.join(WD, 'X.npy'))
    shutil.copy2(INPUT_DATA_y, WD)
    shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii'))
    #################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    prob_class1 = np.count_nonzero(y) / float(len(y))

    SEED = 23071991
    cv = [[tr.tolist(), te.tolist()]
            for tr, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS,
              shuffle=True, random_state=SEED)]
    cv.insert(0, None)  # first fold is None

    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    INPUT_MASK = 'mask.nii'
        # parameters grid
    # Re-run with
    l1 = np.array([[0], [0.3], [0.7], [1]])
    l1l2tv = np.hstack([l1, l1 - l1, 1 - l1])
    alphas = [.01, .05, .1, .5, 1.]
    k_ratio = 1
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]] * \
                                                   l1l2tv.shape[0]),
                                        l1l2tv] for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv,
                                         np.array([[k_ratio]] * \
                                                 alphal1l2tv.shape[0])]])
    params = [params.tolist() for params in alphal1l2tvk]
    user_func_filename = os.path.join(os.environ["HOME"], "git",
        "scripts", "2014_deptms", "03_enettv_config_maskdep.py")
    print "user_func", user_func_filename
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  dilatation_pocess=dilat_process,
                  structure=INPUT_MASK,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results.csv",
                  penalty_start=penalty_start,
                  prob_class1=prob_class1)
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)

    """######################################################################
    print "# Start by running Locally with 2 cores, to check that everything is OK)"
    print "mapreduce.py --map %s/config.json --ncore 2" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# 1) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr'
    print "# 2) Run one Job to test"
    print "qsub -I"
    print "cd %s" % WD_CLUSTER
    print "./job_Global_long.pbs"
    print "# 3) Run on cluster"
    print "qsub job_Global_long.pbs"
    print "# 4) Log out and pull Pull"
    print "exit"
    print sync_pull_filename
    #########################################################################
    print "# Reduce"
    print "mapreduce.py --reduce %s/config.json" % WD"""