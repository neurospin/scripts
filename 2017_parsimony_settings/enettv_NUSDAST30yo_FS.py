#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:11:02 2017

@author: ad247405
"""


import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.linalgs import LinearOperatorNesterov
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from statsmodels.stats.inter_rater import fleiss_kappa
from parsimony.algorithms.utils import Info
from parsimony.algorithms.utils import AlgorithmSnapshot
import parsimony.utils.weights as weights

WD = '/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer'
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")

def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")

penalty_start = 3


##############################################################################
def init():
    INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/mask.npy'
    INPUT_LINEAR_OPE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/Atv.npz'

    os.makedirs(WD, exist_ok=True)
    shutil.copy(INPUT_DATA_X, WD)
    shutil.copy(INPUT_DATA_y, WD)
    shutil.copy(INPUT_MASK_PATH, WD)
    shutil.copy(INPUT_LINEAR_OPE_PATH, WD)

    ## Create config file
    y = np.load(INPUT_DATA_y)
    X = np.load(INPUT_DATA_X)

    #start_vector=weights.RandomUniformWeights(normalise=True,seed= 40004)
    #np.save("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/start_vector",start_vector)
    start_vector = np.load("/neurospin/brainomics/2017_parsimony_settings/NUSDAST_30yo/Freesurfer/start_vector.npy")

    params = [[0.01,0.72,0.08,0.2],
             [0.01,0.08,0.72,0.2],
             [0.01,0.18,0.02,0.8],
             [0.1,0.18,0.02,0.8],
             [0.1,0.02,0.18,0.8],
             [0.01,0.02,0.18,0.8],
             [0.1,0.08,0.72,0.2],
             [0.1,0.72,0.08,0.2]]


    assert len(params) == 8

    user_func_filename = "/home/ad247405/git/scripts/2017_parsimony_settings/enettv_NUSDAST30yo_FS.py"

    config = dict(data=dict(X="X.npy", y="y.npy",start_vector = "start_vector.npy"),
                  params=params,
                  structure="mask.npy",
                  structure_linear_operator_tv="Atv.npz",
                  map_output="model_selectionCV",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="model_selectionCV.csv")
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))


    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, _ = \
        clust_utils.gabriel_make_sync_data_files(WD, wd_cluster=WD_CLUSTER)
    cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd,walltime = "2500:00:00")


#############################################################################
def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.A = LinearOperatorNesterov(filename=config["structure_linear_operator_tv"])
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def mapper(key, output_collector):
    import mapreduce as GLOBAL
    X = GLOBAL.DATA["X"]
    y = GLOBAL.DATA["y"]
    start_vector = GLOBAL.DATA["start_vector"]

    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))

    class_weight="auto" # unbiased
    mask = np.ones(X.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    A = GLOBAL.A

    info = [Info.converged,Info.num_iter,Info.time,Info.func_val,Info.mu,Info.gap]
    conesta = algorithms.proximal.CONESTA()
    algorithm_params = dict(max_iter=1000000, info=info)
    out = os.path.join(WD,"model_selectionCV","0",str(key[0])+"_"+ str(key[1]) + "_" +\
                                                      str(key[2]) +"_"+str(key[3]),"conesta_ite_snapshots/")
    os.makedirs(out, exist_ok=True)

    snapshot = AlgorithmSnapshot(out, saving_period=1).save_conesta
    algorithm_params["callback"] = snapshot



    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,\
                                             algorithm_params=algorithm_params,\
                                             class_weight=class_weight,\
                                             penalty_start=penalty_start,start_vector=start_vector)
    mod.fit(X, y.ravel())
    y_pred = mod.predict(X)
    proba_pred = mod.predict_probability(X)
    ret = dict(y_pred=y_pred, y_true=y, proba_pred=proba_pred, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret