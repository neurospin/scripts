#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Tue Jun 13 16:33:17 CEST 2017

@author: edouard.duchesnay@cea.fr
"""

import os
import json
import numpy as np
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from parsimony.utils.linalgs import LinearOperatorNesterov
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from statsmodels.stats.inter_rater import fleiss_kappa

WD = "/neurospin/brainomics/2016_deptms/analysis/Freesurfer/enetall_deptms-FS"
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")
WD_ORIGINAL = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/results/deptms_FS_enettv_10000ite'
DATA_PATH = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data'
user_func_filename = "/home/ed203246/git/scripts/2016_deptms/Freesurfer_scripts/03_enetall_deptms-FS.py"

def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,os.path.basename(WD) + "_dcv.xlsx")

NFOLDS_OUTER = 5
NFOLDS_INNER = 5
penalty_start = 2
DATA_TYPE = "mesh"
#DATA_TYPE = "image"

lambda_max_A = 8.999

##############################################################################
def init():

    os.makedirs(WD, exist_ok=True)
    shutil.copy(os.path.join(DATA_PATH, 'X.npy'), WD)
    shutil.copy(os.path.join(DATA_PATH, 'y.npy'), WD)
    shutil.copy(os.path.join(DATA_PATH, 'mask.nii'), WD)

    if DATA_TYPE == "image":
        shutil.copy(os.path.join(DATA_PATH, 'mask.nii'), WD)
    elif DATA_TYPE == "mesh":
        shutil.copy(os.path.join(DATA_PATH, 'mask.npy'), WD)
        shutil.copy(os.path.join(DATA_PATH, 'lrh.pial.gii'), WD)

    shutil.copy(os.path.join(DATA_PATH, "Atv.npz"), WD)
    #shutil.copy(INPUT_LINEAR_OPE_PATH, WD)

    ## Create config file
    os.chdir(WD)
    X = np.load("X.npy")
    y = np.load("y.npy")

    ## Create config file

    #  ########################################################################
    #  Setting 1: 5cv + large range of parameters: cv_largerange
    #  with sub-sample training set with size 50, 100
    # 5cv/cv0*[_sub50]/refit/*

    # sub_sizes = [50, 100]
    sub_sizes = []

    cv_outer = [[tr, te] for tr, te in
                StratifiedKFold(n_splits=NFOLDS_OUTER, random_state=42).split(np.zeros(y.shape[0]), y.ravel())]

    # check we got the same CV than previoulsy
    cv_old = json.load(open(os.path.join(WD_ORIGINAL, "config_dCV.json")))["resample"]
    cv_outer_old = [cv_old[k] for k in ['cv%02d/refit' % i for i in  range(NFOLDS_OUTER)]]
    assert np.all([np.all(np.array(cv_outer_old[i][0]) == cv_outer[i][0]) for i in range(NFOLDS_OUTER)])
    assert np.all([np.all(np.array(cv_outer_old[i][1]) == cv_outer[i][1]) for i in range(NFOLDS_OUTER)])
    # check END

    import collections
    cv = collections.OrderedDict()

    cv["refit/refit"] = [np.arange(len(y)), np.arange(len(y))]

    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        # Simple CV
        cv["cv%02d/refit" % (cv_outer_i)] = [tr_val, te]

        # Nested CV
        # cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
        # for cv_inner_i, (tr, val) in enumerate(cv_inner):
        #     cv["cv%02d/cvnested%02d" % ((cv_outer_i), cv_inner_i)] = [tr_val[tr], tr_val[val]]

        # Sub-sample training set with size 50, 100
        # => cv*_sub[50|100]/refit
        grps = np.unique(y[tr_val]).astype(int)
        ytr = y.copy()
        ytr[te] = np.nan
        g_idx = [np.where(ytr == g)[0] for g in grps]
        assert np.all([np.all(ytr[g_idx[g]] == g) for g in grps])

        g_size = np.array([len(g) for g in g_idx])
        g_prop = g_size / g_size.sum()

        for sub_size in sub_sizes:
            # sub_size = sub_sizes[0]
            sub_g_size = np.round(g_prop * sub_size).astype(int)
            g_sub_idx = [np.random.choice(g_idx[g], sub_g_size[g], replace=False) for g in grps]
            assert np.all([np.all(y[g_sub_idx[g]] == g) for g in grps])
            tr_val_sub = np.concatenate(g_sub_idx)
            assert len(tr_val_sub) == sub_size
            assert np.all([idx in tr_val for idx in tr_val_sub])
            assert np.all(np.logical_not([idx in te for idx in tr_val_sub]))
            cv["cv%02d_sub%i/refit" % (cv_outer_i, sub_size)] = [tr_val_sub, te]

    cv = {k:[cv[k][0].tolist(), cv[k][1].tolist()] for k in cv}

    # Nested CV
    # assert len(cv_largerange) == NFOLDS_OUTER * NFOLDS_INNER + NFOLDS_OUTER + 1

    # Simple CV
    # assert len(cv) == NFOLDS_OUTER + 1

    # Simple CV + sub-sample training set with size 50, 100:
    assert len(cv) == NFOLDS_OUTER * (1 + len(sub_sizes)) + 1

    print(list(cv.keys()))

    # Large grid of parameters
    alphas = [0.01, 0.1, 1.0]
    # alphas = [.01, 0.1, 1.0] # first ran with this grid
    tv_ratio = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    l1l2_ratio = [0.1, 0.5, 0.9]
    #l1l2_ratio = [0, 0.1, 0.5, 0.9, 1.0] # first ran with this grid
    algos = ["enettv", "enetgn"]
    params_enet_tvgn = [list(param) for param in itertools.product(algos, alphas, l1l2_ratio, tv_ratio)]
    assert len(params_enet_tvgn) == 198

    params_enet = [list(param) for param in itertools.product(["enet"], alphas, l1l2_ratio, [0])]
    assert len(params_enet) ==  9 # old 15

    params = params_enet_tvgn + params_enet
    assert len(params) == 207
    # Simple CV
    # assert len(params) * len(cv) == 1890

    # Simple CV + sub-sample training set with size 50, 100:
    assert len(params) * len(cv) == 1242

    config = dict(data=dict(X="X.npy", y="y.npy"),
                  params=params, resample=cv,
                  structure_linear_operator_tv="Atv.npz",
                  map_output="5cv",
                  user_func=user_func_filename)
    json.dump(config, open(os.path.join(WD, "config_cv_largerange.json"), "w"))


    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    cmd = "mapreduce.py --map  %s/config_cv_largerange.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd,walltime = "250:00:00",
                                            suffix="_cv_largerange",
                                            freecores=2)

    #  ########################################################################
    #  Setting 2: dcv + reduced range of parameters: dcv_reducedrange
    #  5cv/cv0*/cvnested0*/*

    cv_outer = [[tr, te] for tr, te in
                StratifiedKFold(n_splits=NFOLDS_OUTER, random_state=42).split(np.zeros(y.shape[0]), y.ravel())]

    # check we got the same CV than previoulsy
    cv_old = json.load(open(os.path.join(WD_ORIGINAL, "config_dCV.json")))["resample"]
    cv_outer_old = [cv_old[k] for k in ['cv%02d/refit' % i for i in  range(NFOLDS_OUTER)]]
    assert np.all([np.all(np.array(cv_outer_old[i][0]) == cv_outer[i][0]) for i in range(NFOLDS_OUTER)])
    assert np.all([np.all(np.array(cv_outer_old[i][1]) == cv_outer[i][1]) for i in range(NFOLDS_OUTER)])
    # check END

    import collections
    cv = collections.OrderedDict()
    cv["refit/refit"] = [np.arange(len(y)), np.arange(len(y))]

    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        cv["cv%02d/refit" % (cv_outer_i)] = [tr_val, te]
        cv_inner = StratifiedKFold(n_splits=NFOLDS_INNER, random_state=42).split(np.zeros(y[tr_val].shape[0]), y[tr_val].ravel())
        for cv_inner_i, (tr, val) in enumerate(cv_inner):
            cv["cv%02d/cvnested%02d" % ((cv_outer_i), cv_inner_i)] = [tr_val[tr], tr_val[val]]

    cv = {k:[cv[k][0].tolist(), cv[k][1].tolist()] for k in cv}
    #assert len(cv) == NFOLDS_OUTER + 1
    assert len(cv) == NFOLDS_OUTER * NFOLDS_INNER + NFOLDS_OUTER + 1
    print(list(cv.keys()))

    # Reduced grid of parameters
    alphas = [0.001, 0.01, 0.1, 1.0]
    tv_ratio = [0.2, 0.8]
    l1l2_ratio = [0.1, 0.9]
    algos = ["enettv", "enetgn"]
    params_enet_tvgn = [list(param) for param in itertools.product(algos, alphas, l1l2_ratio, tv_ratio)]
    assert len(params_enet_tvgn) == 32 # 16

    params_enet = [list(param) for param in itertools.product(["enet"], alphas, l1l2_ratio, [0])]
    assert len(params_enet) == 8 # 4

    params = params_enet_tvgn + params_enet
    assert len(params) == 40 # 20
    assert len(params) * len(cv) == 1240 # 620

    config = dict(data=dict(X="X.npy", y="y.npy"),
                  params=params, resample=cv,
                  structure_linear_operator_tv="Atv.npz",
                  beta_start="beta_start.npz",
                  map_output="5cv",
                  user_func=user_func_filename)
    json.dump(config, open(os.path.join(WD, "config_dcv_reducedrange.json"), "w"))

    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    cmd = "mapreduce.py --map  %s/config_dcv_reducedrange.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd,walltime = "250:00:00",
                                            suffix="_dcv_reducedrange",
                                            freecores=2)


#############################################################################
def load_globals(config):
    import scipy.sparse as sparse
    import functools
    import mapreduce as GLOBAL  # access to global variables

    GLOBAL.DATA = GLOBAL.load_data(config["data"])

    Atv = LinearOperatorNesterov(filename=config["structure_linear_operator_tv"])
    Agn = sparse.vstack(Atv)
    Agn.singular_values = Atv.get_singular_values()
    def get_singular_values(self, nb=None):
        return self.singular_values[nb] if nb is not None else self.singular_values
    Agn.get_singular_values = functools.partial(get_singular_values, Agn)
    assert np.allclose(Agn.get_singular_values(0), lambda_max_A, rtol=1e-03, atol=1e-03)
    GLOBAL.Atv, GLOBAL.Agn = Atv, Agn

    # npz = np.load(config["beta_start"])
    # GLOBAL.beta_start = {k:npz[k] for k in npz}

def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    """
    # debug mapper
    config = json.load(open(os.path.join(WD, "config_cv_largerange.json"), "r"))
    load_globals(config)
    resample(config, 'refit/refit')
    key = ('enettv', 0.01, 0.1, 0.3)
    """
    import mapreduce as GLOBAL
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]

    # key = 'enettv_0.01_0.1_0.2'.split("_")
    algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])

    tv = alpha * tvratio
    l1 = alpha * float(1 - tv) * l1l2ratio
    l2 = alpha * float(1 - tv) * (1- l1l2ratio)

    print(key, algo, alpha, l1, l2, tv)
    # alpha = float(key[0])
    # l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    # print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))

    class_weight = "auto"  # unbiased

    # beta_start = GLOBAL.beta_start["lambda_%.4f" % alpha]
    # mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)
    if algo == 'enettv':
        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        mod = estimators.LogisticRegressionL1L2TV(l1, l2, tv,  GLOBAL.Atv,
            algorithm=conesta, class_weight=class_weight, penalty_start=penalty_start)
    elif algo == 'enetgn':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.LogisticRegressionL1L2GraphNet(l1, l2, tv, GLOBAL.Agn,
            algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)
    elif algo == 'enet':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.ElasticNetLogisticRegression(l1l2ratio, alpha,
            algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)
    else:
        raise Exception('Algo%s not handled' %algo)

    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred, beta=mod.beta)#, mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

#############################################################################
# Reducer
try:
    REDUCER_SRC = '/home/ed203246/git/brainomics-team/2017_logistic_nestv/scripts/reduce_plot_vizu.py'
    exec(open(REDUCER_SRC).read())
except:
    pass

def do_reducer():
    output_filename = results_filename()
    # reducer(WD=WD, output_filename=output_filename, force_recompute=False)
    model =  estimators.LogisticRegression()
    # reducer(WD=WD, output_filename=output_filename, force_recompute=True, model=model, rescale = False)
    # reducer(WD=WD, output_filename=output_filename, force_recompute=False, model=model, rescale=False)
    reducer(WD=WD, output_filename=output_filename, force_recompute=False, model=model, rescale=True)


###############################################################################
# copy old results to new organization
import glob

def dir_from_param_list(param_list):
    return "_".join([str(p) for p in param_list])


def param_src_to_dst(param_src, dst_prefix=None, dst_suffix=None):
    a, l1, l2, tv = [float(a) for a in param_src.split("_")]
    try:
        l1_ratio = np.round(l1 / (l1+l2), 5)
    except:
        l1_ratio = np.nan
    param_list_dst = [a, l1_ratio, tv]
    if dst_prefix is not None:
        param_list_dst = [dst_prefix] + param_list_dst
    if dst_suffix is not None:
        param_list_dst = param_list_dst + [dst_suffix]
    return dir_from_param_list(param_list_dst)


def sync(SRC, DST, outer_str, inner_str, dst_prefix, copy=True):
    for fold in itertools.product(outer_str, inner_str):
        # fold = ('cv00', 'cvnested00')
        src = SRC % fold
        # src = SRC % (fold[0], fold[1].replace('refit', 'all'))
        dst = DST % fold
        for param_src in [os.path.basename(p) for p in glob.glob(src + "/*")]:
            #param_src = '0.01_0.07_0.63_0.3'
            #param_src = "0.01_0.02_0.18_0.8_-1"
            #param_src = "0.1_0.72_0.08_0.2_-1"
            # param_src = "0.01_0.16_0.64_0.2"
            param_dst = param_src_to_dst(param_src, dst_prefix=dst_prefix)
            path_src = os.path.join(src, param_src, "beta.npz")
            path_dst = os.path.join(dst, param_dst, "beta.npz")
            if os.path.exists(path_src) and os.path.exists(path_dst):
#                cor = None
#                beta_src = np.load(path_src)['arr_0']
#                beta_dst = np.load(path_dst)['arr_0']
#                cor = np.corrcoef(beta_dst.ravel(), beta_src.ravel())[0, 1]
#                print(fold, "\t", param_src, "\t", param_dst,  "\t",cor,  "\t", beta_dst.shape)
                print("RM", os.path.dirname(path_dst))
                shutil.rmtree(os.path.dirname(path_dst))
            elif os.path.exists(path_dst):
                beta_dst = np.load(path_dst)['arr_0']
                print(fold, "\t", param_dst, "\t", beta_dst.shape)
            elif copy and os.path.exists(path_src) and not os.path.exists(path_dst):
                print(path_src, "\n", path_dst)
                # Data in SRC ARE WRONG DO NOT COPY
#                shutil.copytree(
#                        os.path.dirname(path_src),
#                        os.path.dirname(path_dst))

def do_sync():
    outer_str = ["cv%02d" % i for i in range(5)]
    inner_str = ["cvnested%02d" % i for i in range(5)] + ["refit"]
    SRC = os.path.join(WD_ORIGINAL, "model_selectionCV/%s/%s")
    DST = os.path.join(WD, "5cv/%s/%s")
    # Data in SRC ARE WRONG DO NOT COPY
    # sync(SRC, DST, outer_str, inner_str, dst_prefix="enettv")
    sync(SRC, DST, outer_str, inner_str, dst_prefix="enettv", copy=False)
