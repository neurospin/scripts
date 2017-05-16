# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:24:57 2016

@author: ad247405
"""



import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
import brainomics.image_atlas
import parsimony.algorithms as algorithms
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
from collections import OrderedDict
import pandas as pd
from parsimony.utils.linalgs import LinearOperatorNesterov
import shutil

WD = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/results/enettv_10000ite'
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")
NFOLDS_OUTER = 5
NFOLDS_INNER = 5
penalty_start = 3
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = np.load(config["structure"])
    A = LinearOperatorNesterov(filename=config["structure_linear_operator_tv"])
    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE


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

    penalty_start = 2

    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))

    class_weight="auto" # unbiased

    mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)
    A = GLOBAL.A

    conesta = algorithms.proximal.CONESTA(max_iter=10000)
    #conesta = algorithms.proximal.CONESTA(max_iter=500)
    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,class_weight=class_weight,penalty_start=penalty_start)
    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret



def scores(key, paths, config, as_dataframe=False):
    import mapreduce
    print(key)
    assert len(paths) == NFOLDS_INNER or len(paths) == NFOLDS_OUTER, "Failed for key %s" % key
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = [item["proba_pred"].ravel() for item in values]
    prob_pred = np.concatenate(prob_pred)

    # Prediction performances
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred) #area under curve score.

    # P-values
    success = r * s
    success = success.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pvalue_recall0_true_prob = binom_test(success[0], s[0], 1 - prob_class1,alternative = 'greater')
    pvalue_recall1_true_prob = binom_test(success[1], s[1], prob_class1,alternative = 'greater')
    pvalue_recall0_unknwon_prob = binom_test(success[0], s[0], 0.5,alternative = 'greater')
    pvalue_recall1_unknown_prob = binom_test(success[1], s[1], 0.5,alternative = 'greater')
    pvalue_recall_mean = binom_test(success[0]+success[1], s[0] + s[1], p=0.5,alternative = 'greater')


    # Beta's measures of similarity
    betas = np.hstack([item["beta"][penalty_start:, :] for item in values]).T

    # Correlation
    R = np.corrcoef(betas)
    #print R
    R = R[np.triu_indices_from(R, 1)]
    # Fisher z-transformation / average
    z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
    # bracktransform
    r_bar = (np.exp(2 * z_bar) - 1) /  (np.exp(2 * z_bar) + 1)

    # threshold betas to compute fleiss_kappa and DICE
    try:
        betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in range(betas.shape[0])])
        #print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
        #print(np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5,
        #                   rtol=0, atol=1e-02))

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

    scores = OrderedDict()
    scores['key'] = key
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
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores["auc"] = auc
    scores['pvalue_recall0_true_prob_one_sided'] = pvalue_recall0_true_prob
    scores['pvalue_recall1_true_prob_one_sided'] = pvalue_recall1_true_prob
    scores['pvalue_recall0_unknwon_prob_one_sided'] = pvalue_recall0_unknwon_prob
    scores['pvalue_recall1_unknown_prob_one_sided'] = pvalue_recall1_unknown_prob
    scores['pvalue_recall_mean'] = pvalue_recall_mean
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas.shape))
    scores['beta_r_bar'] = r_bar
    scores['beta_fleiss_kappa'] = fleiss_kappa_stat
    scores['beta_dice_bar'] = dice_bar

    scores['beta_dice'] = str(dices)
    scores['beta_r'] = str(R)

    if as_dataframe:
        scores = pd.DataFrame([list(scores.values())], columns=list(scores.keys()))

    return scores


def reducer(key, values):
    import os, glob, pandas as pd
    os.chdir(os.path.dirname(config_filename()))
    config = json.load(open(config_filename()))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    param_config_set = set([mapreduce.dir_from_param_list(p) for p in config['params']])
    assert len(paths) / len(param_config_set) == len(config['resample']), "Nb run per param is not the one excpected"
    paths.sort()
    # assert len(paths) == 6603

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="key", score="recall_mean"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    print('## Refit scores: cv*/refit/*')
    print('## -------------------------')
    byparams = groupby_paths([p for p in paths if not p.count("cvnested") and not p.count("refit/refit") ], 3)
    byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)

    print('## doublecv scores by outer-cv and by params: cv*/cvnested*/*')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested") and not p.count("refit/cvnested")  ], 1)
    for fold, paths_fold in bycv.items():
        print(fold)
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}
        data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
    scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)
    assert np.all(np.array([g.shape[0] for d, g in scores_dcv_byparams.groupby('fold')]) == 213)

    # Different settings

    l1l2tv_all = scores_dcv_byparams

    l1l2tv_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1) | close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1) | close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams.tv, 0.2) | close(scores_dcv_byparams.tv, 0.8))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2tv_reduced.groupby('fold')]) == 12)
    assert l1l2tv_reduced.shape[0] == 60

    l1l2tv_ridge_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1) | close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1)) &
        (close(scores_dcv_byparams.tv, 0.2) | close(scores_dcv_byparams.tv, 0.8))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2tv_ridge_reduced.groupby('fold')]) == 6)
    assert l1l2tv_ridge_reduced.shape[0] == 30

    l1l2tv_lasso_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1) | close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams.tv, 0.2) | close(scores_dcv_byparams.tv, 0.8))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2tv_lasso_reduced.groupby('fold')]) == 6)
    assert l1l2tv_lasso_reduced.shape[0] == 30

    l1l2_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)| close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1) | close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams.tv, 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_reduced.groupby('fold')]) == 6)
    assert l1l2_reduced.shape[0] == 30

    l1l2_ridge_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)| close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.1)) &
        (close(scores_dcv_byparams.tv, 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_ridge_reduced.groupby('fold')]) == 3)
    assert l1l2_ridge_reduced.shape[0] == 15

    l1l2_lasso_reduced = scores_dcv_byparams[
        (close(scores_dcv_byparams.a, 0.01) | close(scores_dcv_byparams.a, 0.1)| close(scores_dcv_byparams.a, 1.00)) &
        (close(scores_dcv_byparams.l1_ratio, 0.9)) &
        (close(scores_dcv_byparams.tv, 0))]
    assert np.all(np.array([g.shape[0] for d, g in l1l2_lasso_reduced.groupby('fold')]) == 3)
    assert l1l2_lasso_reduced.shape[0] == 15

    print('## Model selection')
    print('## ---------------')
    l1l2tv_all = argmaxscore_bygroup(l1l2tv_all); l1l2tv_all["method"] = "l1l2tv_all"

    l1l2tv_reduced = argmaxscore_bygroup(l1l2tv_reduced); l1l2tv_reduced["method"] = "l1l2tv_reduced"

    l1l2tv_ridge_reduced = argmaxscore_bygroup(l1l2tv_ridge_reduced); l1l2tv_ridge_reduced["method"] = "l1l2tv_ridge_reduced"

    l1l2tv_lasso_reduced = argmaxscore_bygroup(l1l2tv_lasso_reduced); l1l2tv_lasso_reduced["method"] = "l1l2tv_lasso_reduced"

    l1l2_reduced = argmaxscore_bygroup(l1l2_reduced); l1l2_reduced["method"] = "l1l2_reduced"

    l1l2_ridge_reduced = argmaxscore_bygroup(l1l2_ridge_reduced); l1l2_ridge_reduced["method"] = "l1l2_ridge_reduced"

    l1l2_lasso_reduced = argmaxscore_bygroup(l1l2_lasso_reduced); l1l2_lasso_reduced["method"] = "l1l2_lasso_reduced"

    scores_argmax_byfold = pd.concat([l1l2tv_all,
                                      l1l2tv_reduced, l1l2_reduced,
                                      l1l2tv_ridge_reduced, l1l2_ridge_reduced,
                                      l1l2tv_lasso_reduced, l1l2_lasso_reduced])

    print('## Apply best model on refited')
    print('## ---------------------------')
    l1l2tv_all = scores("l1l2tv_all",
                               [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                   for index, row in l1l2tv_all.iterrows()],
                                config, as_dataframe=True)

    l1l2tv_reduced = scores("l1l2tv_reduced",
                            [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                for index, row in l1l2tv_reduced.iterrows()],
                             config, as_dataframe=True)

    l1l2tv_ridge_reduced = scores("l1l2tv_ridge_reduced",
                                   [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                       for index, row in l1l2tv_ridge_reduced.iterrows()],
                                    config, as_dataframe=True)

    l1l2tv_lasso_reduced = scores("l1l2tv_lasso_reduced",
                                   [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                       for index, row in l1l2tv_lasso_reduced.iterrows()],
                                    config, as_dataframe=True)

    l1l2_reduced = scores("l1l2_reduced",
                                 [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                     for index, row in l1l2_reduced.iterrows()],
                                 config, as_dataframe=True)

    l1l2_ridge_reduced = scores("l1l2_ridge_reduced",
                                 [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                     for index, row in l1l2_ridge_reduced.iterrows()],
                                 config, as_dataframe=True)

    l1l2_lasso_reduced = scores("l1l2_lasso_reduced",
                                 [os.path.join(config['map_output'], row["fold"], "refit", row["key"])
                                     for index, row in l1l2_lasso_reduced.iterrows()],
                                 config, as_dataframe=True)

    scores_cv = pd.concat([l1l2tv_all,
                           l1l2tv_reduced, l1l2_reduced,
                           l1l2tv_ridge_reduced, l1l2_ridge_reduced,
                           l1l2tv_lasso_reduced, l1l2_lasso_reduced,
                           ])

    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='cv_by_param', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='cv_cv_byparam', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='cv_argmax', index=False)
        scores_cv.to_excel(writer, sheet_name='dcv', index=False)

##############################################################################

if __name__ == "__main__":
    WD = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/results/enettv_10000ite'
    INPUT_DATA_X = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/mask.npy'
    INPUT_CSV = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/population.csv'
    INPUT_LINEAR_OPE_PATH = '/neurospin/brainomics/2016_deptms/analysis/Freesurfer/data/Atv.npz'

    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]
    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5


    os.makedirs(WD, exist_ok=True)
    shutil.copy(INPUT_DATA_X, WD)
    shutil.copy(INPUT_DATA_y, WD)
    shutil.copy(INPUT_MASK_PATH, WD)
    shutil.copy(INPUT_LINEAR_OPE_PATH, WD)
    ##########################################
    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)

    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
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

        else:
            cv["cv%02d/refit" % (cv_outer_i -1)] = [tr_val, te]
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]


    print(list(cv.keys()))

    tv_range = tv_ratios = [0.0,.1,0.2,0.3, 0.4,0.5,.6,0.7, .8,0.9,1.0]
    ratios = np.array([[.5, .5, 1], [.1, .90, 1],[0.9,0.1,1]])
    alphas = [.1,.01,1.0]

    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])

    params = [np.round(params,2).tolist() for params in alphal1l2tv]


    user_func_filename = "/home/ad247405/git/scripts/2016_deptms/Freesurfer_scripts/03_enettv_model_selection.py"

    config = dict(data=dict(X=os.path.basename(INPUT_DATA_X), y=os.path.basename(INPUT_DATA_y)),
                  params=params, resample=cv,
                  structure=os.path.basename(INPUT_MASK_PATH),
                  structure_linear_operator_tv="Atv.npz",
                  map_output="model_selectionCV",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="model_selectionCV.csv")
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))


        # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
