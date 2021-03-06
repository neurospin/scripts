# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:54:55 2014

@author: cp243490
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from parsimony.algorithms.utils import Info
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from scipy.stats import binom_test
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
from collections import OrderedDict

import shutil

NFOLDS = 5


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
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
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight = "auto"  # unbiased
    alpha = float(key[0])
    l1, l2 = alpha * float(key[1]), alpha * float(key[2])
    tv, k_ratio = alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k_ratio:%f" % (l1, l2, tv, k_ratio)

    n_voxels = np.count_nonzero(STRUCTURE.get_data())
    if k_ratio != -1:
        k = n_voxels * k_ratio
        k = int(k)
        aov = SelectKBest(k=k)
        aov.fit(Xtr[..., penalty_start:], ytr.ravel())
        mask = STRUCTURE.get_data() != 0
        mask[mask] = aov.get_support()
        #print mask.sum()
        A, _ = tv_helper.A_from_mask(mask)
        Xtr_r = np.hstack([Xtr[:, :penalty_start],
                           Xtr[:, penalty_start:][:, aov.get_support()]])
        Xte_r = np.hstack([Xte[:, :penalty_start],
                           Xte[:, penalty_start:][:, aov.get_support()]])
    else:
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
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    # DEBUG
    import mapreduce as GLOBAL
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

if __name__ == "__main__":

    #########################################################################
    ## load data
    BASE_PATH = WD = "/neurospin/brainomics/2014_imagen_fu2_adrs/"

    DATASET_PATH = os.path.join(BASE_PATH,    "ADRS_datasets")
    WD = os.path.join(BASE_PATH,              "ADRS_enettv_discrete-adrs")
    if not os.path.exists(WD):
        os.makedirs(WD)

    penalty_start = 3

    #########################################################################
    ## Build config file for all couple (Modality, roi)
    print "ok1"
    INPUT_DATA_X = os.path.join(DATASET_PATH, 'X_discrete-adrs.npy')
    INPUT_DATA_y = os.path.join(DATASET_PATH, 'y_discrete-adrs.npy')
    INPUT_MASK = os.path.join(DATASET_PATH,
                              'mask_atlas_binarized_discrete-adrs.nii.gz')
    # copy X, y, mask file names in the current directory
#    shutil.copy2(INPUT_DATA_X, os.path.join(WD, 'X.npy'))
    shutil.copy2(INPUT_DATA_y, WD)
    shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii.gz'))
    print "ok1"
    #################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)

    cv = [[tr.tolist(), te.tolist()]
            for tr, te in StratifiedKFold(y.ravel(), n_folds=NFOLDS,
              shuffle=True, random_state=0)]
    prob_class1 = np.count_nonzero(y) / float(len(y))

    INPUT_DATA_X = 'X.npy'
    INPUT_DATA_y = 'y.npy'
    INPUT_MASK = 'mask.nii.gz'
    # parameters grid
    # Re-run with

    tv_range = np.hstack([np.arange(0, 1., .1), [0.05, 0.01, 0.005, 0.001]])
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],
                       [.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])
    alphas = [.01, .05, .1, .5, 1., 10.]
    #k_range = [100, 1000, 10000, 100000, -1]
    k_range = [-1]
    l1l2tv = [np.array([[float(1 - tv),
                        float(1 - tv), tv]]) * ratios for tv in tv_range]
    l1l2tv.append(np.array([[0., 0., 1.]]))
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]] * l1l2tv.shape[0]),
                                        l1l2tv] \
                                  for alpha in alphas])
    alphal1l2tvk = np.concatenate([np.c_[alphal1l2tv,
                                      np.array([[k]] * alphal1l2tv.shape[0])] \
                                  for k in k_range])
    params = [params.tolist() for params in alphal1l2tvk]

    user_func_filename = os.path.abspath(__file__)
    print "user_func", user_func_filename
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="ADRS_discrete.csv",
                  penalty_start=penalty_start)
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
#    # Sync to cluster
#    print "Sync data to gabriel.intra.cea.fr: "
#    os.system(sync_push_filename)

    ######################################################################
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
    print "mapreduce.py --reduce %s/config.json" % WD