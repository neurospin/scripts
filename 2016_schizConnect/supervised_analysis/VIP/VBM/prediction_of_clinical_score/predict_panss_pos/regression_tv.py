#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:21:46 CEST 2017

@author: amicie.depierrefeu@cea.fr
"""

import os
import json
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn import metrics
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
import scipy
from sklearn import linear_model
import parsimony.algorithms.gradient as gradient

WD = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/prediction_of_clinical_score/predict_panss_pos'
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")

DATA_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_panss'
user_func_filename = '/home/ad247405/git/scripts/2016_schizConnect/supervised_analysis/VIP/VBM/prediction_of_clinical_score/predict_panss_pos/regression_tv.py'

def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,os.path.basename(WD) + "_dcv.xlsx")

NFOLDS_OUTER = 5
NFOLDS_INNER = 5
#DATA_TYPE = "mesh"
DATA_TYPE = "image"

penalty_start = 1
##############################################################################
def init():
    os.makedirs(WD, exist_ok=True)
    shutil.copy(os.path.join(DATA_PATH, 'X_scz_panss_pos.npy'), WD)
    shutil.copy(os.path.join(DATA_PATH, 'panss_pos.npy'), WD)
    # VBM
    if DATA_TYPE == "image":
        shutil.copy(os.path.join(DATA_PATH, 'mask.nii'), WD)
    elif DATA_TYPE == "mesh":
        shutil.copy(os.path.join(DATA_PATH, 'mask.npy'), WD)
        shutil.copy(os.path.join(DATA_PATH, 'lrh.pial.gii'), WD)

    shutil.copy(os.path.join(DATA_PATH, "Atv.npz"), WD)



    ## Create config file
    os.chdir(WD)
    X = np.load("X_scz_panss_pos.npy")
    y = np.load("panss_pos.npy")

    X = X[np.logical_not(np.isnan(y)).ravel(),:]
    y = y[np.logical_not(np.isnan(y))]
    assert X.shape == (36, 125960)


    ## Create config file
    cv_outer = [[tr, te] for tr,te in KFold(n=len(y), n_folds=NFOLDS_OUTER, random_state=42)]
    if cv_outer[0] is not None: # Make sure first fold is None
        cv_outer.insert(0, None)
        null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
        cv_outer[0] = null_resampling

#

    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        if cv_outer_i == 0:
            cv["refit/refit"] = [tr_val, te]
            cv_inner = KFold(n=len(tr_val), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["refit/cvnested%02d" % (cv_inner_i)] = [tr_val[tr], tr_val[val]]
        else:
            cv["cv%02d/refit" % (cv_outer_i -1)] = [tr_val, te]
            cv_inner = KFold(n= len(tr_val), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]


    print(list(cv.keys()))

    # Large grid of parameters
    alphas = [0.01, 0.01, 0.1, 1.0]
    # alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    tv_ratio = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    l1l2_ratio = [0.1, 0.5, 0.9]
    #l1l2_ratio = [0, 0.1, 0.5, 0.9, 1.0]
    algos = ["enettv", "enetgn"]
    params_enet_tvgn = [list(param) for param in itertools.product(algos, alphas, l1l2_ratio, tv_ratio)]
    assert len(params_enet_tvgn) == 264

    params_enet = [list(param) for param in itertools.product(["enet"], alphas, l1l2_ratio, [0])]
    assert len(params_enet) == 12

    params_ridge = [list(param) for param in itertools.product(["Ridge"], alphas, l1l2_ratio, [0])]
    assert len(params_ridge) == 12

    params = params_enet_tvgn + params_enet + params_ridge
    assert len(params) == 288

    # Simple CV + sub-sample training set with size 50, 100:
    assert len(params) * len(cv) == 10368

    params = [['linearSklearn', 0.01, 0.1, 0.01]]

    config = dict(data=dict(X="X_scz_panss_pos.npy", y="panss_pos.npy"),
                  params=params , resample=cv,
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
    GLOBAL.Atv, GLOBAL.Agn = Atv, Agn


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

    # key = 'enettv_0.01_0.1_0.2'.split("_")
    algo, alpha, l1l2ratio, tvratio = key[0], float(key[1]), float(key[2]), float(key[3])

    tv = alpha * tvratio
    l1 = alpha * float(1 - tv) * l1l2ratio
    l2 = alpha * float(1 - tv) * (1- l1l2ratio)

    print(key, algo, alpha, l1, l2, tv)

    scaler = preprocessing.StandardScaler().fit(Xtr[:,1:])
    Xtr[:,1:] = scaler.transform(Xtr[:,1:])
    Xte[:,1:] = scaler.transform(Xte[:,1:])


    if algo == 'enettv':
        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        mod = estimators.LinearRegressionL1L2TV(l1, l2, tv,  GLOBAL.Atv,
            algorithm=conesta,penalty_start = penalty_start )
        mod.fit(Xtr, ytr.ravel())
        beta = mod.beta

    elif algo == 'enetgn':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.LinearRegressionL1L2GraphNet(l1, l2, tv, GLOBAL.Agn,
            algorithm=fista,penalty_start = penalty_start )
        mod.fit(Xtr, ytr.ravel())
        beta=mod.beta

    elif algo == 'enet':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.ElasticNet(l1l2ratio,
            algorithm=fista,penalty_start = penalty_start )
        mod.fit(Xtr, ytr.ravel())
        beta=mod.beta

    elif algo == 'Ridge':
        mod = estimators.RidgeRegression(l1l2ratio,penalty_start = penalty_start)
        mod.fit(Xtr, ytr.ravel())
        beta=mod.beta

    elif algo == 'RidgeAGD':
        mod = estimators.RidgeRegression(l1l2ratio,\
        algorithm=gradient.GradientDescent(max_iter=1000),penalty_start = penalty_start )
        mod.fit(Xtr, ytr.ravel())
        beta=mod.beta


    elif algo == 'linearSklearn':
        mod = linear_model.LinearRegression(fit_intercept=False)
        mod.fit(Xtr, ytr.ravel())
        beta=mod.coef_
        beta = beta.reshape(beta.shape[0],1)

    elif algo == 'SkRidge':
        mod = linear_model.Ridge(alpha = l1l2ratio,fit_intercept=False)
        mod.fit(Xtr, ytr.ravel())
        beta=mod.coef_
        beta = beta.reshape(beta.shape[0],1)

    elif algo == 'SkRidgeInt':
        mod = linear_model.Ridge(alpha = l1l2ratio,fit_intercept=True)
        mod.fit(Xtr, ytr.ravel())
        beta=mod.coef_
        beta = beta.reshape(beta.shape[0],1)
    else:
        raise Exception('Algo%s not handled' %algo)

    y_pred = mod.predict(Xte)
    ret = dict(y_pred=y_pred, y_true=yte,beta=beta)
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
    reducer(WD=WD, output_filename=output_filename, force_recompute=True, model=model, rescale=True)


def scores(key, paths, config, as_dataframe=False, algo_idx=None):
    key_parts = key.split("_")
    algo = key_parts[algo_idx] if algo_idx is not None else None
    key_parts.remove(algo)
    if len(key_parts) > 0:
        try:
            params = [float(p) for p in key_parts]
        except:
            params = [None, None, None]
    print(algo, params)
    if (len(paths) != NFOLDS_INNER) or (len(paths) != NFOLDS_OUTER):
        print("Failed for key %s" % key)
        return None

    values = [mapreduce.OutputCollector(p) for p in paths]
    try:
        values = [item.load() for item in values]
    except Exception as e:
        print(e)
        return None

    y_true_splits = [item["y_true"].ravel() for item in values]
    y_pred_splits = [item["y_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true_splits)
    y_pred = np.concatenate(y_pred_splits)

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_true, y_pred)
    r_squared = metrics.r2_score(y_true, y_pred)
    betas = np.hstack([item["beta"] for item in values]).T
    # threshold betas to compute fleiss_kappa and DICE
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in range(betas.shape[0])])
    #Compute pvalue
    scores = OrderedDict()
    scores['key'] = key
    scores['algo'] = algo
    scores['a'], scores['l1_ratio'], scores['tv_ratio'] = params
    scores['algo'] = algo
    try:
        a, l1, l2 , tv  = [float(par) for par in key.split("_")]
        scores['a'] = a
        scores['l1'] = l1
        scores['l2'] = l2
        scores['tv'] = tv
        left = float(1 - tv)
        if left == 0: left = 1.
        scores['l1_ratio'] = float(l1) / left
    except:
        pass
    scores['slope'] = slope
    scores['intercept'] = intercept
    scores['r_squared'] = r_squared
    scores['r_value'] = r_value
    scores['r_value_absolute_value'] = np.abs(r_value)
    scores['p_value'] = p_value
    scores['std_err'] = std_err
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas.shape))
    scores['param_key'] = key
    if as_dataframe:
        scores = pd.DataFrame([list(scores.values())], columns=list(scores.keys()))

    return scores


def reducer(key=None, values=None):

    import os, glob, pandas as pd
    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def scores_groupby_paths(paths, param_pos, algo_pos_in_params, score_func):
        byparams = groupby_paths(paths, param_pos)
        # key='enettv_0.1_0.1_0.2'; paths=byparams[key]; algo_idx=algo_pos_in_params
        byparams_scores = {k:score_func(k, v, config, algo_idx=algo_pos_in_params) for k, v in byparams.items()}
        byparams_scores = {k: v for k, v in byparams_scores.items() if v is not None}
        data = [list(byparams_scores[k].values()) for k in byparams_scores]
        columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
        return pd.DataFrame(data, columns=columns)

    def argmaxscore_bygroup(data, groupby='fold', param_key="key",
                            score="r_squared",
                            refit_key=None,  # Do refit ?
                            config=None,   # required for refit
                            score_func=None,   # required for refit
                            algo_pos_in_params=None   # required for refit
                            ):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        arg_max_byfold = pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])
        arg_max_byfold["key_refit"] = refit_key
        if refit_key is not None:
            refit = score_func(refit_key,
                                    [os.path.join(config['map_output'], row[groupby], "refit", row[param_key])
                                        for index, row in arg_max_byfold.iterrows()],
                                     config, as_dataframe=True, algo_idx=algo_pos_in_params)
        else:
            refit = None

        return arg_max_byfold, refit


    # config_cv_largerange
    s = 'tv_ratio'
    os.chdir(WD)
    config = json.load(open("config_cv_largerange.json"))
    paths_all = glob.glob("5cv/cv0?/refit/*")
    paths_all.sort()

    #assert len(paths) == 4286
    print('## Refit scores: cv*/refit/*')
    print('## -------------------------')
    scores_refit = scores_groupby_paths(paths=paths_all, param_pos=3, algo_pos_in_params=0, score_func=scores)


    print('## doublecv scores by outer-cv and by params: cv*/cvnested*/*')
    print('## -----------------------------------------')
    paths = glob.glob("5cv/cv0?/cvnested0?/*")
    paths.sort()
    bycv = groupby_paths(paths, 1)
    scores_dcv_byparams = None
    for fold, paths_fold in bycv.items():
        print(fold)
        scores_dcv_fold = scores_groupby_paths(paths=paths_fold, param_pos=3, algo_pos_in_params=0, score_func=scores)
        scores_dcv_fold["fold"] = fold
        scores_dcv_byparams = pd.concat([scores_dcv_byparams, scores_dcv_fold])

    print([[g, d.shape[0]] for g, d in scores_dcv_byparams.groupby(["fold", "algo"])])
    # assert np.all(np.array([g.shape[0] for d, g in scores_dcv_byparams.groupby('fold')]) == 136)

    # Different settings
    results = list()

    for algo in ["enettv", "enetgn"]:
        # algo = "enettv"
        results.append(argmaxscore_bygroup(data=scores_dcv_byparams[scores_dcv_byparams.algo == algo],
            refit_key="%s_dcv-all" % algo, config=config, score_func=scores, algo_pos_in_params=0))

        l1l2s_reduced = scores_dcv_byparams[
            (scores_dcv_byparams.algo == algo) &
            (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
            (close(scores_dcv_byparams.l1_ratio, 0.1) | close(scores_dcv_byparams.l1_ratio, 0.9)) &
            (close(scores_dcv_byparams[s], 0.2) | close(scores_dcv_byparams[s], 0.8))]
        # assert np.all(np.array([g.shape[0] for d, g in l1l2s_reduced.groupby('fold')]) == 8)
        # assert l1l2s_reduced.shape[0] == 40
        results.append(argmaxscore_bygroup(data=l1l2s_reduced,
            refit_key="%s_dcv-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

        l1l2s_ridge_reduced = scores_dcv_byparams[
            (scores_dcv_byparams.algo == algo) &
            (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
            (close(scores_dcv_byparams.l1_ratio, 0.1)) &
            (close(scores_dcv_byparams[s], 0.2) | close(scores_dcv_byparams[s], 0.8))]
        # assert np.all(np.array([g.shape[0] for d, g in l1l2s_ridge_reduced.groupby('fold')]) == 4)
        # assert l1l2s_ridge_reduced.shape[0] == 20
        results.append(argmaxscore_bygroup(data=l1l2s_ridge_reduced,
            refit_key="%s_dcv-ridge-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

        l1l2s_ridge_reduced2 = l1l2s_ridge_reduced[close(l1l2s_ridge_reduced[s], 0.8)] # FS 0.8
        results.append(argmaxscore_bygroup(data=l1l2s_ridge_reduced2,
            refit_key="%s_dcv-ridge-reduced2" % algo, config=config, score_func=scores, algo_pos_in_params=0))

        l1l2s_lasso_reduced = scores_dcv_byparams[
            (scores_dcv_byparams.algo == algo) &
            (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
            (close(scores_dcv_byparams.l1_ratio, 0.9)) &
            (close(scores_dcv_byparams[s], 0.2) | close(scores_dcv_byparams[s], 0.8))]
        # assert np.all(np.array([g.shape[0] for d, g in l1l2s_lasso_reduced.groupby('fold')]) == 4)
        # assert l1l2s_lasso_reduced.shape[0] == 20
        results.append(argmaxscore_bygroup(data=l1l2s_lasso_reduced,
            refit_key="%s_dcv-lasso-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

        l1l2s_lasso_reduced2 = l1l2s_lasso_reduced[close(l1l2s_lasso_reduced[s], 0.8)] # FS 0.8
        results.append(argmaxscore_bygroup(data=l1l2s_lasso_reduced2,
            refit_key="%s_dcv-lasso-reduced2" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    algo = "enet"
    l1l2_reduced = scores_dcv_byparams[
        (scores_dcv_byparams.algo == algo) &
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1) | close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams[s], 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_reduced.groupby('fold')]) == 4)
    assert l1l2_reduced.shape[0] == 20
    results.append(argmaxscore_bygroup(data=l1l2_reduced,
        refit_key="%s_dcv-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    l1l2_ridge_reduced = scores_dcv_byparams[
        (scores_dcv_byparams.algo == algo) &
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1)) &
        (close(scores_dcv_byparams[s], 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_ridge_reduced.groupby('fold')]) == 2)
    assert l1l2_ridge_reduced.shape[0] == 10
    results.append(argmaxscore_bygroup(data=l1l2_ridge_reduced,
        refit_key="%s_dcv-ridge-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    l1l2_lasso_reduced = scores_dcv_byparams[
        (scores_dcv_byparams.algo == algo) &
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)) &
        (close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams[s], 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_lasso_reduced.groupby('fold')]) == 2)
    assert l1l2_lasso_reduced.shape[0] == 10
    results.append(argmaxscore_bygroup(data=l1l2_lasso_reduced,
        refit_key="%s_dcv-lasso-reduced" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    algo = "linearSklearn"

    linearSklearn = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
    results.append(argmaxscore_bygroup(data=linearSklearn,
        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    algo = "Ridge"

    Ridge = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
    results.append(argmaxscore_bygroup(data=Ridge,
        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    algo = "RidgeAGD"

    RidgeAGD = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
    results.append(argmaxscore_bygroup(data=RidgeAGD,
        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))


    algo = "SkRidge"

    SkRidge = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
    results.append(argmaxscore_bygroup(data=SkRidge,
        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    algo = "RidgeMeanFalse"

    RidgeMeanFalse = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
    results.append(argmaxscore_bygroup(data=RidgeMeanFalse,
        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))
#
#    algo = "linearParsimony"
#
#    linear_parsimony = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
#    results.append(argmaxscore_bygroup(data=linear_parsimony,
#        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))
#
#    algo = "linearParsimonyMeanFalse"
#    linearParsimony_meanFalse = scores_dcv_byparams[(scores_dcv_byparams.algo == algo)]
#    results.append(argmaxscore_bygroup(data=linearParsimony_meanFalse,
#        refit_key="%s_dcv" % algo, config=config, score_func=scores, algo_pos_in_params=0))

    cv_argmax, scores_dcv = zip(*results)
    cv_argmax = pd.concat(cv_argmax)
    scores_dcv = pd.concat(scores_dcv)

    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='cv_by_param', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='cv_cv_byparam', index=False)
        cv_argmax.to_excel(writer, sheet_name='cv_argmax', index=False)
        scores_dcv.to_excel(writer, sheet_name='dcv', index=False)


###############################################################################
def plot_scores(input_filename, sheetnames, x_col, y_cols, group_by, log_scale,
                colors, linestyle, select, outpout_filename):
    # %matplotlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns

    for sheetname in sheetnames:
        # sheetname = sheetnames[0]
        #os.path.join(os.path.dirname(input_filename),
        #                              sheetname + "_scores-by-s.pdf")
        data = pd.read_excel(input_filename, sheetname=sheetname)

        # avoid poor rounding
        for col in group_by:
            try:
                data[col] = np.asarray(data[col]).round(5)
            except:
                pass
        data[x_col] = np.asarray(data[x_col]).round(5); # assert len(data[x_col].unique()) == 11
        def close(vec, val, tol=1e-4):
            return np.abs(vec - val) < tol

        # select
        for k in select:
            mask_or = np.zeros(data.shape[0], dtype=bool)
            for v in select[k]:
                mask_or = np.logical_or(mask_or, close(data[k], v))
            data = data[mask_or]

        # enet => tv or gn == 0
        enettv0 = data[data["algo"] == "enet"].copy()
        enettv0.algo = "enettv"
        enetgn0 = data[data["algo"] == "enet"].copy()
        enetgn0.algo = "enetgn"
        data = pd.concat([data, enettv0, enetgn0])

        # rm enet
        data = data[~(data.algo == 'enet')]

        data.sort_values(by=x_col, ascending=True, inplace=True)

        pdf = PdfPages(outpout_filename)
        for y_col in y_cols:
            #y_col = y_cols[0]
            fig=plt.figure()
            xoffsset = -0.001 * len([_ for _ in data.groupby(group_by)]) / 2
            for (algo, l1_ratio, a), d in data.groupby(group_by):
                print((algo, l1_ratio, a))
                plt.plot(d[x_col], d[y_col], color=colors[(a, l1_ratio)],
                         ls = linestyle[algo],
                         label="%s, l1/l2:%.1f, a:%.3f" % (algo, l1_ratio, a))
                if y_col in log_scale:
                    plt.yscale('log')
                y_col_se = y_col + "_se"
                if y_col_se in d.columns:
                    plt.errorbar(d[x_col] + xoffsset, d[y_col], yerr=d[y_col_se], legend=False, fmt=None,
                     alpha=0.2, ecolor=colors[(a, l1_ratio)], elinewidth=1)
                    xoffsset += 0.001
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
            plt.suptitle(y_col)
            pdf.savefig(fig); plt.clf()
        pdf.close()

def do_plot():
    import seaborn as sns

    input_filename = results_filename()

    # scores
    y_cols = ['r_value_absolute_value','prop_non_zeros_mean']
    log_scale = ['prop_non_zeros_mean']

    x_col = 'tv_ratio'
    group_by = ["algo", "l1_ratio", "a"]
    sheetnames = ['cv_by_param']

    # palette
    n_colors = 4
    pal = dict(
            reds = sns.color_palette("Reds", n_colors=n_colors),
            blues = sns.color_palette("Blues", n_colors=n_colors),
            greens = sns.color_palette("Greens", n_colors=n_colors))

    # sns.palplot(pal['reds'])
    # sns.palplot(pal['blues'])
    # sns.palplot(pal['greens'])
    # colors mapping
    colors = {
             (0.001, 0.1):pal['blues'][0],
             (0.001, 0.5):pal['greens'][0],
             (0.001, 0.9):pal['reds'][0],
             (0.01, 0.1):pal['blues'][1],
             (0.01, 0.5):pal['greens'][1],
             (0.01, 0.9):pal['reds'][1],
             (0.1, 0.1):pal['blues'][2],
             (0.1, 0.5):pal['greens'][2],
             (0.1, 0.9):pal['reds'][2],
             (1.0, 0.1):pal['blues'][3],
             (1.0, 0.5):pal['greens'][3],
             (1.0, 0.9):pal['reds'][3]}

    linestyle = dict(enettv="-", enetgn="--")

    plot_scores(input_filename, sheetnames, x_col, y_cols, group_by,
                colors, linestyle, log_scale,
                select = dict(l1_ratio = [0.1, 0.5, 0.9], a=[.001, .01, .1, 1.0]),
                outpout_filename=os.path.join(WD,
                                      os.path.basename(WD) +"_"+ sheetnames[0] + "_scores-by-s.pdf"))

    plot_scores(input_filename, sheetnames,
                x_col, y_cols, group_by, log_scale,
                colors, linestyle,
                select=dict(a=[0.001, 0.01, 0.1, 1.0], l1_ratio=[0.1, 0.9]),
                outpout_filename=os.path.join(WD,
                                      os.path.basename(WD) +"_"+ sheetnames[0] + "_scores-by-s_lasso-ridge.pdf"))

    plot_scores(input_filename, sheetnames,
                x_col, y_cols, group_by, log_scale,
                colors, linestyle,
                select=dict(a=[0.001, 0.01, 0.1, 1.0], l1_ratio=[0.1]),
                outpout_filename=os.path.join(WD,
                                      os.path.basename(WD) +"_"+ sheetnames[0] + "_scores-by-s_ridge.pdf"))

###############################################################################
# copy old results to new organization
import glob

def dir_from_param_list(param_list):
    return "_".join([str(p) for p in param_list])


def param_src_to_dst(param_src, dst_prefix=None, dst_suffix=None):
    a, l1, l2, tv = [float(a) for a in param_src.split("_")]
    try:
        l1_ratio = round(l1 / (l1 + l2), 10)
    except ZeroDivisionError:
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
        dst = DST % fold
        for param_src in [os.path.basename(p) for p in glob.glob(src + "/*")]:
            param_dst = param_src_to_dst(param_src, dst_prefix=dst_prefix)
            path_src = os.path.join(src, param_src, "beta.npz")
            path_dst = os.path.join(dst, param_dst, "beta.npz")
            if os.path.exists(path_src) and os.path.exists(path_dst):
                cor = None
                beta_src = np.load(path_src)['arr_0']
                beta_dst = np.load(path_dst)['arr_0']
                cor = np.corrcoef(beta_dst.ravel(), beta_src.ravel())[0, 1]
                print(fold, "\t", param_src, "\t", param_dst,  "\t",cor)
            elif copy and os.path.exists(path_src) and not os.path.exists(path_dst):
                print(path_src, "\n", path_dst)
                shutil.copytree(
                        os.path.dirname(path_src),
                        os.path.dirname(path_dst))

def do_sync():
    outer_str = ["cv%02d" % i for i in range(5)]
    inner_str = ["cvnested%02d" % i for i in range(5)] + ["refit"]

    # copy enetgn
    SRC = os.path.join(WD_ORIGINAL, "enetgn", "enetgn_NUDAST_50yo", "model_selectionCV/%s/%s")
    DST = os.path.join(WD, "5cv/%s/%s")
    sync(SRC, DST, outer_str, inner_str, dst_prefix="enetgn")

    SRC = os.path.join(WD_ORIGINAL, "enettv", "enettv_NUDAST_50yoVBM", "model_selectionCV/%s/%s")
    DST = os.path.join(WD, "5cv/%s/%s")
    sync(SRC, DST, outer_str, inner_str, dst_prefix="enettv", copy=False)