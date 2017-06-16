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
import glob

WD = '/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/FS/no_covariates/warm_restart/cv03_all_as_start_vector'
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")

def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")

penalty_start = 3


##############################################################################
if __name__ == "__main__":
    INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/mask.npy'
    INPUT_LINEAR_OPE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/30yo/Atv.npz'
    NFOLDS_OUTER = 5
    os.makedirs(WD, exist_ok=True)
    shutil.copy(INPUT_DATA_y, WD)
    shutil.copy(INPUT_MASK_PATH, WD)
    shutil.copy(INPUT_LINEAR_OPE_PATH, WD)
    # remove covariates from data
    X = np.load(INPUT_DATA_X)
    np.save(os.path.join(WD,"X.npy"),X[:,penalty_start:])

    if not os.path.exists(os.path.join(WD, "beta_start.npy")):
        betas = dict()
        BETA_START_PATH = "/neurospin/brainomics/2017_parsimony_settings/warm_restart/NUSDAST_30yo/FS/no_covariates/no_warm_restart/model_selectionCV/cv03/all"
        params = glob.glob(os.path.join( BETA_START_PATH,"0*"))
        for p in params:
            print(p)
            path = os.path.join(p,"beta.npz")
            beta = np.load(path)
            betas[os.path.basename(p)] = beta['arr_0']

        np.save(os.path.join(WD, "beta_start.npy"),betas)
        beta_start = np.load(os.path.join(WD, "beta_start.npy"))
        #assert np.all([np.all(beta_start[a] == betas[a]) for a in beta_start.keys()])


    y = np.load(INPUT_DATA_y)

    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]

#
    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
            cv["cv%02d/all" % (cv_outer_i)] = [tr_val, te]

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

    user_func_filename = "/home/ad247405/git/scripts/2017_parsimony_settings/warm_start/no_covariates/warm_start/NUDAST_30yo_FS_cv03_all_as_start_vector.py"

    config = dict(data=dict(X="X.npy", y="y.npy"),
                  params=params,resample=cv,
                  structure="mask.npy",
                  beta_start = "beta_start.npy",
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
    GLOBAL.DIR = config["map_output"]
    GLOBAL.BETA_START = np.load(config["beta_start"])


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

    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))
    print (key)
    class_weight="auto" # unbiased
    print(output_collector.output_dir)
    beta_start = GLOBAL.BETA_START.all()[os.path.basename(output_collector.output_dir)]
    mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)
    A = GLOBAL.A

    info = [Info.converged,Info.num_iter,Info.time,Info.func_val,Info.mu,Info.gap]
    conesta = algorithms.proximal.CONESTA()
    algorithm_params = dict(max_iter=50000, info=info)
    out = os.path.join(WD_CLUSTER,output_collector.output_dir,"conesta_ite_snapshots/")

    os.makedirs(out, exist_ok=True)

    snapshot = AlgorithmSnapshot(out, saving_period=1).save_conesta
    algorithm_params["callback"] = snapshot



    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,\
                                             algorithm_params=algorithm_params,\
                                             class_weight=class_weight)
    mod.fit(Xtr, ytr.ravel(),beta = beta_start)
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred, beta=mod.beta,  mask=mask,beta_start=beta_start)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret
