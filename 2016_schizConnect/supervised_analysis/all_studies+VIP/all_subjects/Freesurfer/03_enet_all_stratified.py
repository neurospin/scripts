#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 09:21:46 CEST 2017

"""

import os
import json
import numpy as np
import itertools
from sklearn.cross_validation import StratifiedKFold
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


WD = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/with_preserved_ratios/enetall'
WD_CLUSTER = WD.replace("/neurospin/", "/mnt/neurospin/sel-poivre/")

DATA_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data'
user_func_filename = '/home/ad247405/git/scripts/2016_schizConnect/supervised_analysis/all_studies+VIP/all_subjects/Freesurfer/03_enet_all_stratified.py'

def config_filename(): return os.path.join(WD,"config_cv_largerange.json")
def results_filename(): return os.path.join(WD,os.path.basename(WD) + "_dcv.xlsx")

NFOLDS_OUTER = 5
NFOLDS_INNER = 5
penalty_start = 3
DATA_TYPE = "mesh"
#DATA_TYPE = "image"


##############################################################################
def init():
    os.makedirs(WD, exist_ok=True)
    shutil.copy(os.path.join(DATA_PATH, 'X.npy'), WD)
    shutil.copy(os.path.join(DATA_PATH, 'y.npy'), WD)

    # VBM
    if DATA_TYPE == "image":
        shutil.copy(os.path.join(DATA_PATH, 'mask.nii'), WD)
    elif DATA_TYPE == "mesh":
        shutil.copy(os.path.join(DATA_PATH, 'mask.npy'), WD)
        shutil.copy(os.path.join(DATA_PATH, 'lrh.pial.gii'), WD)

    shutil.copy(os.path.join(DATA_PATH, "Atv.npz"), WD)



    ## Create config file
    os.chdir(WD)
    X = np.load("X.npy")
    y = np.load("y.npy")


    fold1 = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold1.npy")
    fold2 = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold2.npy")
    fold3 = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold3.npy")
    fold4 = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold4.npy")
    fold5 = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/fold_stratified/fold5.npy")

    ## Create config file
    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
    cv_outer[0][0] = np.concatenate((fold2,fold3,fold4,fold5))
    cv_outer[0][1] = fold1
    cv_outer[1][0] = np.concatenate((fold1,fold3,fold4,fold5))
    cv_outer[1][1] = fold2
    cv_outer[2][0] = np.concatenate((fold1,fold2,fold4,fold5))
    cv_outer[2][1] = fold3
    cv_outer[3][0] = np.concatenate((fold1,fold2,fold3,fold5))
    cv_outer[3][1] = fold4
    cv_outer[4][0] = np.concatenate((fold1,fold2,fold3,fold4))
    cv_outer[4][1] = fold5
#

    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        cv["cv%02d/all" % (cv_outer_i)] = [tr_val, te]
        cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
        for cv_inner_i, (tr, val) in enumerate(cv_inner):
            cv["cv%02d/cvnested%02d" % ((cv_outer_i), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]


    print(list(cv.keys()))


    # Large grid of parameters
    alphas = [0.001, 0.01, 0.1, 1.0]
    # alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    tv_ratio = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    l1l2_ratio = [0.1, 0.5, 0.9]
    #l1l2_ratio = [0, 0.1, 0.5, 0.9, 1.0]
    algos = ["enettv", "enetgn"]
    params_enet_tvgn = [list(param) for param in itertools.product(algos, alphas, l1l2_ratio, tv_ratio)]
    assert len(params_enet_tvgn) == 264

    params_enet = [list(param) for param in itertools.product(["enet"], alphas, l1l2_ratio, [0])]
    assert len(params_enet) == 12

    params = params_enet_tvgn + params_enet
    assert len(params) == 276 #315

    # Simple CV + sub-sample training set with size 50, 100:
    assert len(params) * len(cv) ==  8280

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

    # mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte = scaler.transform(Xte)

    if algo == 'enettv':
        conesta = algorithms.proximal.CONESTA(max_iter=10000)
        mod = estimators.LogisticRegressionL1L2TV(l1, l2, tv,  GLOBAL.Atv,
            algorithm=conesta, class_weight=class_weight, penalty_start=penalty_start)
        mod.fit(Xtr, ytr.ravel())
    elif algo == 'enetgn':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.LogisticRegressionL1L2GraphNet(l1, l2, tv, GLOBAL.Agn,
            algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)
        mod.fit(Xtr, ytr.ravel())
    elif algo == 'enet':
        fista = algorithms.proximal.FISTA(max_iter=5000)
        mod = estimators.ElasticNetLogisticRegression(l1l2ratio, alpha,
            algorithm=fista, class_weight=class_weight, penalty_start=penalty_start)
        mod.fit(Xtr, ytr.ravel())
    else:
        raise Exception('Algo%s not handled' %algo)

    #mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred,beta=mod.beta)
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
    # print(key, paths)
    # key = 'enettv_0.1_0.5_0.1'
    # paths = ['5cv/cv00/refit/enetgn_0.1_0.9_0.1', '5cv/cv01/refit/enetgn_0.1_0.9_0.1', '5cv/cv02/refit/enetgn_0.1_0.9_0.1', '5cv/cv03/refit/enetgn_0.1_0.9_0.1', '5cv/cv04/refit/enetgn_0.1_0.9_0.1']
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
    prob_pred_splits = [item["proba_pred"].ravel() for item in values]
    prob_pred = np.concatenate(prob_pred_splits)

    # Prediction performances
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred)

    # balanced accuracy (recall_mean)
    bacc_splits = [recall_score(y_true_splits[f], y_pred_splits[f], average=None).mean() for f in range(len(y_true_splits))]
    auc_splits = [roc_auc_score(y_true_splits[f], prob_pred_splits[f]) for f in range(len(y_true_splits))]

    print("bacc all - mean(bacc) %.3f" % (r.mean() - np.mean(bacc_splits)))
    # P-values
    success = r * s
    success = success.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pvalue_recall0_true_prob = binom_test(success[0], s[0], 1 - prob_class1,alternative = 'greater')
    pvalue_recall1_true_prob = binom_test(success[1], s[1], prob_class1,alternative = 'greater')
    pvalue_recall0_unknwon_prob = binom_test(success[0], s[0], 0.5,alternative = 'greater')
    pvalue_recall1_unknown_prob = binom_test(success[1], s[1], 0.5,alternative = 'greater')
    pvalue_bacc = binom_test(success[0]+success[1], s[0] + s[1], p=0.5,alternative = 'greater')

    # Beta's measures of similarity
    betas = np.hstack([item["beta"][penalty_start:, :] for item in values]).T

    # Correlation
    R = np.corrcoef(betas)
    R = R[np.triu_indices_from(R, 1)]
    # Fisher z-transformation / average
    z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
    # bracktransform
    r_bar = (np.exp(2 * z_bar) - 1) / (np.exp(2 * z_bar) + 1)

    # threshold betas to compute fleiss_kappa and DICE
    try:
        betas_t = np.vstack([
                array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0]
                for i in range(betas.shape[0])])
        # Compute fleiss kappa statistics
        beta_signed = np.sign(betas_t)
        table = np.zeros((beta_signed.shape[1], 3))
        table[:, 0] = np.sum(beta_signed == 0, 0)
        table[:, 1] = np.sum(beta_signed == 1, 0)
        table[:, 2] = np.sum(beta_signed == -1, 0)
        fleiss_kappa_stat = fleiss_kappa(table)

        # Paire-wise Dice coeficient
        ij = [[i, j] for i in range(betas.shape[0]) for j in range(i+1, betas.shape[0])]
        dices = list()
        for idx in ij:
            A, B = beta_signed[idx[0], :], beta_signed[idx[1], :]
            dices.append(float(np.sum((A == B)[(A != 0) & (B != 0)])) / (np.sum(A != 0) + np.sum(B != 0)))
        dice_bar = np.mean(dices)
    except:
        dice_bar = fleiss_kappa_stat = 0

    # Proportion of selection within the support accross the CV
    support_count = (betas_t != 0).sum(axis=0)
    support_count = support_count[support_count > 0]
    support_prop = support_count / betas_t.shape[0]

    scores = OrderedDict()
    scores['key'] = key
    scores['algo'] = algo
    scores['a'], scores['l1_ratio'], scores['tv_ratio'] = params

    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['bacc'] = r.mean()
    scores['bacc_se'] = np.std(bacc_splits) / np.sqrt(len(bacc_splits))
    scores["auc"] = auc
    scores['auc_se'] = np.std(auc_splits) / np.sqrt(len(auc_splits))
    scores['pvalue_recall0_true_prob_one_sided'] = pvalue_recall0_true_prob
    scores['pvalue_recall1_true_prob_one_sided'] = pvalue_recall1_true_prob
    scores['pvalue_recall0_unknwon_prob_one_sided'] = pvalue_recall0_unknwon_prob
    scores['pvalue_recall1_unknown_prob_one_sided'] = pvalue_recall1_unknown_prob
    scores['pvalue_bacc_mean'] = pvalue_bacc
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas.shape))
    scores['beta_r_bar'] = r_bar
    scores['beta_fleiss_kappa'] = fleiss_kappa_stat
    scores['beta_dice_bar'] = dice_bar
    scores['beta_dice'] = str(dices)
    scores['beta_r'] = str(R)
    scores['beta_support_prop_select_mean'] = support_prop.mean()
    scores['beta_support_prop_select_sd'] = support_prop.std()

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
                            score="bacc",
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

    # config = json.load(open(config_filename()))
    # paths = glob.glob(os.path.join(WD, "5cv", "*", "*", "*"))
    # param_config_set = set([mapreduce.dir_from_param_list(p) for p in config['params']])
    # assert len(paths) / len(param_config_set) == len(config['resample']), "Nb run per param is not the one excpected"

    # config_cv_largerange
    s = 'tv_ratio'
    os.chdir(WD)
    config = json.load(open("config_cv_largerange.json"))
    paths_all = glob.glob("5cv/cv0?/refit/*")
    paths_all.sort()
    # paths_sub50 = glob.glob("5cv/cv0?_sub50/refit/*")
    # paths_sub50.sort()
    # paths_sub100 = glob.glob("5cv/cv0?_sub100/refit/*")
    # paths_sub100.sort()

    #assert len(paths) == 4286
    print('## Refit scores: cv*/refit/*')
    print('## -------------------------')
    scores_refit = scores_groupby_paths(paths=paths_all, param_pos=3, algo_pos_in_params=0, score_func=scores)
    #scores_refit_sub50 = scores_groupby_paths(paths=paths_sub50, param_pos=3, algo_pos_in_params=0, score_func=scores)
    #scores_refit_sub100 = scores_groupby_paths(paths=paths_sub100, param_pos=3, algo_pos_in_params=0, score_func=scores)

    # with pd.ExcelWriter(os.path.join(WD, "results_refit_cv_by_param_largerange.xlsx")) as writer:
    #     scores_refit.to_excel(writer, sheet_name='cv_by_param_all', index=False)
        #scores_refit_sub100.to_excel(writer, sheet_name='cv_by_param_sub100', index=False)
        #scores_refit_sub50.to_excel(writer, sheet_name='cv_by_param_sub50', index=False)


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
        results.append(argmaxscore_bygroup(data=scores_dcv_byparams[
            (scores_dcv_byparams.algo == algo)],
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
    y_cols = ['bacc', 'auc', 'beta_r_bar', 'beta_fleiss_kappa',
              'beta_dice_bar', 'beta_support_prop_select_mean',
              'prop_non_zeros_mean']
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
            #param_src = "0.01_0.008_0.792_0.2_-1"
            #param_src = "0.01_0.02_0.18_0.8_-1"
            #param_src = "0.1_0.72_0.08_0.2_-1"
            # param_src = "0.01_0.16_0.64_0.2"
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
