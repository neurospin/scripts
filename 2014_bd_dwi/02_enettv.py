# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:45:08 2014

@author: christophe

create the config file for the dataset created in 01_build_dataset.py.
"""

import os, sys
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
import shutil
from collections import OrderedDict

import brainomics.cluster_gabriel as clust_utils


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["structure"])
    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE
    GLOBAL.penalty_start = config['penalty_start']


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
        #raise ImportError("could not import ")
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.penalty_start
    class_weight="auto" # unbiased
    alpha = float(key[0])
    l1, l2, tv, k = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k:%i" % (l1, l2, tv, k)
    if k != -1:
        k = int(k)
        aov = SelectKBest(k=k)
        aov.fit(Xtr[..., penalty_start:], ytr.ravel())
        mask = STRUCTURE.get_data() != 0
        mask[mask] = aov.get_support()
        #print mask.sum()
        A, _ = tv_helper.A_from_mask(mask)
        Xtr_r = np.hstack([Xtr[:, :penalty_start], Xtr[:, penalty_start:][:, aov.get_support()]])
        Xte_r = np.hstack([Xte[:, :penalty_start], Xte[:, penalty_start:][:, aov.get_support()]])
    else:
        mask = np.ones(Xtr.shape[0], dtype=bool)
        Xtr_r = Xtr
        Xte_r = Xte
        A = GLOBAL.A
    mod = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight=class_weight)
    mod.fit(Xtr_r, ytr)
    y_pred = mod.predict(Xte_r)
    y_post = mod.predict_probability(Xte_r) # a posteriori probability
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta,  mask=mask,
               y_post=y_post)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret



def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    #import glob, mapreduce
    #values = [mapreduce.OutputCollector(p) for p in glob.glob("/neurospin/brainomics/2014_mlc/GM_UNIV/results/*/0.05_0.45_0.45_0.1_-1.0/")]
    # Compute sd; ie.: compute results on each folds

    # Avoid taking into account the fold 0
    values = [item.load() for item in values[1:]]

    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    y_post = np.concatenate([item["y_post"].ravel() for item in values])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = None
    auc = roc_auc_score(y_true, y_post) #area under curve score.
    a, l1, l2 , tv , k = [float(par) for par in key.split("_")]
    scores = OrderedDict()
    scores['key'] = key
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
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
    scores['n_ite'] = n_ite
    scores['auc'] = auc
    scores['a'] = a
    scores['l1'] = l1
    scores['l2'] = l2
    scores['k'] = k
    scores['tv'] = tv

    return scores


##############################################################################
## Run all
def run_all(config):
    import mapreduce
    WD = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi"
    key = '_'.join(str(p) for p in config['params'][0])
    #class GLOBAL: DATA = dict()
    load_globals(config)
    OUTPUT = os.path.join(WD, 'test', key)
    # run /home/ed203246/bin/mapreduce.py
    oc = mapreduce.OutputCollector(OUTPUT)
    #if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
    X = np.load(os.path.join(WD, 'X.npy'))
    y = np.load(os.path.join(WD, 'Y.npy'))
    mapreduce.DATA_RESAMPLED = {}
    mapreduce.DATA_RESAMPLED["X"] = [X, X]
    mapreduce.DATA_RESAMPLED["y"] = [y, y]
    params = np.array([float(p) for p in key.split("_")])
    mapper(params, oc)
    #oc.collect(key=key, value=ret)

if __name__ == "__main__":

    # Relative filenames
    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'Y.npy'
    INPUT_MASK = "mask_sk.nii.gz"

    # Directory
    INPUT_DIR = "/neurospin/brainomics/2014_bd_dwi/bd_dwi"
    WD = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi"

    # Output
    OUTPUT_CONFIG_FILE = "config.json"
    OUTPUT_LOCAL_CONFIG_FILE = os.path.join(WD, OUTPUT_CONFIG_FILE)

    NFOLDS = 5
    INPUT_PENALTY_START = 3

    #####################
    # Common parameters #
    #####################

    # Resamplings
    y = np.load(os.path.join(INPUT_DIR, INPUT_DATA_y))
    skf = StratifiedKFold(y.ravel(),
                          n_folds=NFOLDS)
    cv = [[tr.tolist(), te.tolist()] for tr,te in skf]
    cv.insert(0, None)  # first fold is None

    # Parameters grid
    tv_range = np.hstack([np.arange(0, 1., .1), [0.05, 0.01, 0.005, 0.001]])
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])
    alphas = [.01, .05, .1 , .5, 1.]
    k_range = [100, 1000, 10000, 100000, -1]
    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv.append(np.array([[0., 0., 1.]]))
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv, np.array([[k]]*alphal1l2tv.shape[0])] for k in k_range])
    params = [params.tolist() for params in alphal1l2tvk]

    # User map/reduce function file:
    user_func_filename = os.path.abspath(sys.argv[0])
    print "user_func", user_func_filename

    ######################
    # Create config file #
    ######################

    # Copy input data
    # Since we use copytree, the target directory should not exist
    if os.path.exists(WD):
        raise IOError("Directory %s exists" % WD)
    shutil.copytree(INPUT_DIR, WD)

    # Config file
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK,
                  penalty_start=INPUT_PENALTY_START,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(OUTPUT_LOCAL_CONFIG_FILE, "w"))

    # Utils files
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD, user="md238665")
    CLUSTER_OUTPUT_FILE = os.path.join(WD_CLUSTER, OUTPUT_CONFIG_FILE)
    cmd = "mapreduce.py --map  %s" % CLUSTER_OUTPUT_FILE
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)

    del INPUT_DIR, WD, INPUT_PENALTY_START
