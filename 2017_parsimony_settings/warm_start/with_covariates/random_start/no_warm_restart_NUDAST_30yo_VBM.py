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

WD = '/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_warm_restart'
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")

def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")

penalty_start = 3


##############################################################################
def init():
    INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/mask.nii'
    INPUT_LINEAR_OPE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/data_30yo/Atv.npz'
    INPUT_START_VECTOR = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_warm_restart/start_vector.npy"
    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5
    os.makedirs(WD, exist_ok=True)
    shutil.copy(INPUT_DATA_X, WD)
    shutil.copy(INPUT_DATA_y, WD)
    shutil.copy(INPUT_MASK_PATH, WD)
    shutil.copy(INPUT_LINEAR_OPE_PATH, WD)
    shutil.copy(INPUT_START_VECTOR, WD)

    #start_vector=weights.RandomUniformWeights(normalise=True,seed= 40004)
    #np.save("/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/VBM/no_warm_restart/start_vector.npy",start_vector)

    y = np.load(INPUT_DATA_y)

    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
    if cv_outer[0] is not None: # Make sure first fold is None
        cv_outer.insert(0, None)
        null_resampling = list();
        null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
        cv_outer[0] = null_resampling

#
    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        if cv_outer_i == 0:
            cv["all/all"] = [tr_val, te]

        else:
            cv["cv%02d/all" % (cv_outer_i -1)] = [tr_val, te]
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]


    print(list(cv.keys()))




    params = [[0.01,0.72,0.08,0.2],
             [0.01,0.08,0.72,0.2],
             [0.01,0.18,0.02,0.8],
             [0.1,0.18,0.02,0.8],
             [0.1,0.02,0.18,0.8],
             [0.01,0.02,0.18,0.8],
             [0.1,0.08,0.72,0.2],
             [0.1,0.72,0.08,0.2]]


    assert len(params) == 8

    user_func_filename = "/home/ad247405/git/scripts/2017_parsimony_settings/warm_restart/no_warm_restart_NUDAST_30yo_VBM.py"

    config = dict(data=dict(X="X.npy", y="y.npy"),
                  params=params,resample=cv,
                  structure="mask.nii",
                  start_vector =dict(start_vector = "start_vector.npy"),
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
    GLOBAL.START_VECTOR = GLOBAL.load_data(config["start_vector"])
    GLOBAL.A = LinearOperatorNesterov(filename=config["structure_linear_operator_tv"])
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.DIR = config["map_output"]

def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    start_vector = GLOBAL.START_VECTOR

    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))

    class_weight="auto" # unbiased
    mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)
    A = GLOBAL.A

    info = [Info.converged,Info.num_iter,Info.time,Info.func_val,Info.mu,Info.gap]
    conesta = algorithms.proximal.CONESTA()
    algorithm_params = dict(max_iter=50000, info=info)
    out = os.path.join(WD,output_collector.output_dir,"conesta_ite_snapshots/")

    os.makedirs(out, exist_ok=True)

    snapshot = AlgorithmSnapshot(out, saving_period=1).save_conesta
    algorithm_params["callback"] = snapshot



    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,\
                                             algorithm_params=algorithm_params,\
                                             class_weight=class_weight,\
                                             penalty_start=penalty_start,start_vector=start_vector)
    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret