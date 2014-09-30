# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 13:45:08 2014

@author: christophe

Create the config file for several datasets and contains the map and reduce
functions.
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
    MASK = nibabel.load(config["mask"])
    A, _ = tv_helper.A_from_mask(MASK.get_data())
    GLOBAL.A, GLOBAL.MASK = A, MASK
    GLOBAL.PENALTY_START = config['penalty_start']


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
    mask = GLOBAL.MASK
    #alpha, ratio_l1, ratio_l2, ratio_tv, k = key
    #key = np.array(key)
    penalty_start = GLOBAL.PENALTY_START
    class_weight="auto" # unbiased
    alpha = float(key[0])
    l1, l2, tv, k = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3]), key[4]
    print "l1:%f, l2:%f, tv:%f, k:%i" % (l1, l2, tv, k)
    if k != -1:
        k = int(k)
        aov = SelectKBest(k=k)
        aov.fit(Xtr[..., penalty_start:], ytr.ravel())
        mask = mask.get_data() != 0
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
    # load return dict correspondning to mapper output. they need to be loaded.
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

    scores = OrderedDict()
    scores['a'] = key[0]
    scores['l1'] = key[1]
    scores['l2'] = key[2]
    scores['tv'] = key[3]
    scores['k'] = key[4]
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

    def create_mapreduce_config(input_x, input_x_desc, input_y, input_mask,
                                resamplings, parameters, penalty_start,
                                output_dir):
        # Create output dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Copy dataset and mask
        shutil.copy2(input_x, output_dir)
        shutil.copy2(input_x_desc, output_dir)
        shutil.copy2(input_y, output_dir)
        shutil.copy2(input_mask, output_dir)

        # Config file
        input_x = os.path.basename(input_x)
        input_y = os.path.basename(input_y)
        input_mask = os.path.basename(input_mask)
        config = dict(data=dict(X=input_x, y=input_y),
                                params=params, resample=resamplings,
                                mask=input_mask,
                                penalty_start=penalty_start,
                                map_output="results",
                                user_func=user_func_filename,
                                reduce_group_by="params",
                                reduce_output="results.csv")
        config_full_filename = os.path.join(output_dir, CONFIG_FILENAME)
        json.dump(config, open(config_full_filename, "w"))

        # Utils files
        sync_push_filename, sync_pull_filename, wd_cluster = \
            clust_utils.gabriel_make_sync_data_files(output_dir,
                                                     user="md238665")
        cluster_output_file = os.path.join(wd_cluster, CONFIG_FILENAME)
        cmd = "mapreduce.py --map  %s" % cluster_output_file
        clust_utils.gabriel_make_qsub_job_files(output_dir, cmd)

    # Directory
    INPUT_DATSETS_DIR = "/neurospin/brainomics/2014_bd_dwi/datasets"
    INPUT_DATA_y = os.path.join(INPUT_DATSETS_DIR, "Y.npy")

    CONFIG_FILENAME = "config.json"

    #####################
    # Common parameters #
    #####################

    # Resamplings
    NFOLDS = 5
    SEED = 13031981
    y = np.load(os.path.join(INPUT_DATSETS_DIR, INPUT_DATA_y))
    skf = StratifiedKFold(y.ravel(),
                          n_folds=NFOLDS,
                          shuffle=True,
                          random_state=SEED)
    resamplings = [[tr.tolist(), te.tolist()] for tr,te in skf]
    resamplings.insert(0, None)  # first fold is None

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

    #######################
    # Create config files #
    #######################

#    # First dataset
#    # This case is no longer interesting: it's keep here for record.
#    input_x = os.path.join(INPUT_DATSETS_DIR, "X_nointercept.npy")
#    input_x_desc = input_x.replace('.npy', '.txt')
#    input_mask = os.path.join(INPUT_DATSETS_DIR, "mask.nii.gz")
#    penalty_start = 2
#    output_dir = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_nointercept"
#    create_mapreduce_config(input_x, input_x_desc, INPUT_DATA_y, input_mask,
#                            resamplings, params, penalty_start,
#                            output_dir)
#    del input_x, input_mask, penalty_start, output_dir

#    # Second dataset
#    # This case is no longer interesting: it's keep here for record.
#    input_x = os.path.join(INPUT_DATSETS_DIR, "X.npy")
#    input_x_desc = input_x.replace('.npy', '.txt')
#    input_mask = os.path.join(INPUT_DATSETS_DIR, "mask.nii.gz")
#    penalty_start = 3
#    output_dir = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi"
#    create_mapreduce_config(input_x, input_x_desc, INPUT_DATA_y, input_mask,
#                            resamplings, params, penalty_start,
#                            output_dir)
#    del input_x, input_mask, penalty_start, output_dir

    # Third dataset
    input_x = os.path.join(INPUT_DATSETS_DIR, "X_site.npy")
    input_x_desc = input_x.replace('.npy', '.txt')
    input_mask = os.path.join(INPUT_DATSETS_DIR, "mask.nii.gz")
    penalty_start = 6
    output_dir = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_site"
    create_mapreduce_config(input_x, input_x_desc, INPUT_DATA_y, input_mask,
                            resamplings, params, penalty_start,
                            output_dir)
    del input_x, input_mask, penalty_start, output_dir

    # Fourth dataset
    input_x = os.path.join(INPUT_DATSETS_DIR, "X_trunc.npy")
    input_x_desc = input_x.replace('.npy', '.txt')
    input_mask = os.path.join(INPUT_DATSETS_DIR, "mask_trunc.nii.gz")
    penalty_start = 6
    output_dir = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_trunc"
    create_mapreduce_config(input_x, input_x_desc, INPUT_DATA_y, input_mask,
                            resamplings, params, penalty_start,
                            output_dir)
    del input_x, input_mask, penalty_start, output_dir

    # Fifth dataset
    input_x = os.path.join(INPUT_DATSETS_DIR, "X_skel.npy")
    input_x_desc = input_x.replace('.npy', '.txt')
    input_mask = os.path.join(INPUT_DATSETS_DIR, "mask_skel.nii.gz")
    penalty_start = 6
    output_dir = "/neurospin/brainomics/2014_bd_dwi/enettv_bd_dwi_skel"
    create_mapreduce_config(input_x, input_x_desc, INPUT_DATA_y, input_mask,
                            resamplings, params, penalty_start,
                            output_dir)
    del input_x, input_mask, penalty_start, output_dir
