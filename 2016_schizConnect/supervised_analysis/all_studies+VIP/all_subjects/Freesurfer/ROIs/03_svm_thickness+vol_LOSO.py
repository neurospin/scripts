#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 10:04:13 2017

@author: ad247405
"""


import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import svm
import pandas as pd
import shutil
from brainomics import array_utils
import mapreduce
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_fscore_support
from statsmodels.stats.inter_rater import fleiss_kappa


WD = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/thick+vol/svm_LOSO'
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")

NFOLDS_OUTER = 5
NFOLDS_INNER = 5
penalty_start = 3
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


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


    c = float(key[0])
    print("c:%f" % (c))

    class_weight='balanced' # unbiased

    mask = np.ones(Xtr.shape[0], dtype=bool)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)

    mod = svm.LinearSVC(C=c,fit_intercept=False,class_weight= class_weight)

    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    y_proba_pred = mod.decision_function(Xte)
    ret = dict(y_pred=y_pred, y_true=yte,prob_pred = y_proba_pred, beta=mod.coef_,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def scores(key, paths, config):
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
    prob_pred_splits = [item["prob_pred"].ravel() for item in values]
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
    betas = np.hstack([item["beta"][:, penalty_start:].T for item in values]).T
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


    return scores

def reducer(key, values):
    import os, glob, pandas as pd
    os.chdir(os.path.dirname(config_filename()))
    config = json.load(open(config_filename()))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if not p.count("0.8_-1")]

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="key", score="bacc"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
#            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    print('## Refit scores')
    print('## ------------')
    byparams = groupby_paths([p for p in paths if p.count("all") and not p.count("all/all")],3)
    byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)

    print('## doublecv scores by outer-cv and by params')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")],1)
    for fold, paths_fold in bycv.items():
        print(fold)
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}
        data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
        scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)


    print('## Model selection')
    print('## ---------------')
    svm = argmaxscore_bygroup(scores_dcv_byparams); svm["method"] = "svm"

    scores_argmax_byfold = svm

    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_svm = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "all", row["key"]) for index, row in svm.iterrows()], config)


    scores_cv = pd.DataFrame([["svm"] + list(scores_svm.values())], columns=["method"] + list(scores_svm.keys()))

    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='cv_by_param', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='cv_cv_byparam', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='cv_argmax', index=False)
        scores_cv.to_excel(writer, sheet_name='dcv', index=False)

##############################################################################

if __name__ == "__main__":
    INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/Xrois_thick+vol+cov.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/y.npy'

    NFOLDS_OUTER = 4 #number of sites
    NFOLDS_INNER = 5


    shutil.copy(INPUT_DATA_X, WD)
    shutil.copy(INPUT_DATA_y, WD)
    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    #COBRE = 1, NMORPH = 2, NUSDAST = 3, VIP = 4
    site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy")


    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
    cv_outer[0][0] = np.transpose(np.where(site != 1)).ravel()
    cv_outer[0][1] = np.transpose(np.where(site == 1)).ravel()  #TEST ON COBRE

    cv_outer[1][0] = np.transpose(np.where(site != 2)).ravel()
    cv_outer[1][1] = np.transpose(np.where(site == 2)).ravel()   # TEST ON NMORPHch

    cv_outer[2][0] = np.transpose(np.where(site != 3)).ravel()
    cv_outer[2][1] = np.transpose(np.where(site == 3)).ravel()  #TEST ON NUSDAST

    cv_outer[3][0] = np.transpose(np.where(site != 4)).ravel()
    cv_outer[3][1] = np.transpose(np.where(site == 4)).ravel() #TEST ON VIP

    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        cv["cv%02d/all" % (cv_outer_i)] = [tr_val, te]
        cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
        for cv_inner_i, (tr, val) in enumerate(cv_inner):
            cv["cv%02d/cvnested%02d" % ((cv_outer_i), cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]


    C_range = [[100],[10],[1],[1e-1],[1e-2],[1e-3],[1e-4],[1e-5],[1e-6],[1e-7],[1e-8],[1e-9]]


    user_func_filename = "/home/ad247405/git/scripts/2016_schizConnect/supervised_analysis/all_studies+VIP/all_subjects/Freesurfer/ROIs/03_svm_thickness+vol_LOSO.py"

    config = dict(data=dict(X="Xrois_thick+vol+cov.npy", y="y.npy"),
                  params=C_range, resample=cv,
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
    clust_utils.gabriel_make_qsub_job_files(WD, cmd,walltime = "150:00:00")