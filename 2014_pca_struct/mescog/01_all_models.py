# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:08:39 2014

@author: md238665

Process mescog/proj_wmh_patterns datasets with standard PCA, sklearn SparsePCA
and our structured PCA.
We use several values for global penalization, TV ratio and L1 ratio.

We generate 2 map_reduce configuration directory (for configuration files and
the files needed to run on the cluster):
 - one for 2 folds CV (close to the protocol in mescog/proj_wmh_patterns)
 - one for 5 folds CV + full resample
Due to the cluster setup and the way mapreduce work we need to copy the dataset
on the cluster. Therefore we copy them for each config directory.
The reducer need the number of folds and if a full resample was used. Therefore
those parameters are included in the config file.

This file was copied from scripts/2014_pca_struct/dice5/01_all_models.py.

"""

import os
import json
import time
import shutil

from itertools import product
from collections import OrderedDict

import numpy as np
import scipy
import pandas as pd

import sklearn.decomposition
from sklearn.cross_validation import StratifiedKFold

import nibabel

import parsimony.functions.nesterov.tv
import pca_tv
import metrics
from brainomics import array_utils
from statsmodels.stats.inter_rater import fleiss_kappa

import brainomics.cluster_gabriel as clust_utils

##################
# Input & output #
##################

INPUT_DIR = os.path.join("/neurospin/",
                         "mescog", "proj_wmh_patterns")

INPUT_DATASET = os.path.join(INPUT_DIR,
                             "X.npy")
INPUT_MASK = os.path.join(INPUT_DIR,
                          "mask_bin.nii")
INPUT_CSV = os.path.join(INPUT_DIR,
                         "population.csv")

OUTPUT_DIR = os.path.join("/neurospin", "brainomics",
                          "2014_pca_struct", "mescog")

##############
# Parameters #
##############

RANDOM_STATE = 13031981
# Parameters for the function create_config
# Note that value at index 1 will be the name of the task on the cluster
CONFIGS = [[2, "mescog_2folds", "config_2folds.json", False],
           [5, "mescog_5folds", "config_5folds.json", True]]

N_COMP = 3
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1])
# Relative penalties
# 0.33 ensures that there is a case with TV = L1 = L2
TVRATIO = np.array([1, 0.5, 0.33, 1e-1, 1e-2, 1e-3, 0])
L1RATIO = np.array([1, 0.5, 1e-1, 1e-2, 1e-3, 0])

PCA_PARAMS = [('pca', 0.0, 0.0, 0.0)]
STRUCT_PCA_PARAMS = list(product(['struct_pca'],
                                 GLOBAL_PENALTIES,
                                 TVRATIO,
                                 L1RATIO))

PARAMS = PCA_PARAMS + STRUCT_PCA_PARAMS

#############
# Functions #
#############


def load_globals(config):
    import mapreduce as GLOBAL
    babel_mask = nibabel.load(config["mask_file"])
    Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_mask(
                                                        babel_mask.get_data())
    GLOBAL.Atv = Atv
    GLOBAL.MASK = babel_mask
    GLOBAL.N_FOLDS = config['n_folds']
    GLOBAL.FULL_RESAMPLE = config['full_resample']


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
    import mapreduce as GLOBAL  # access to global variables:
    # GLOBAL.DATA
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    model_name, global_pen, tv_ratio, l1_ratio = key
    if model_name == 'pca':
        # Force the key
        global_pen = tv_ratio = l1_ratio = 0
    if model_name == 'sparse_pca':
        # Force the key
        tv_ratio = 0
        l1_ratio = 1
        ll1 = global_pen
    if model_name == 'struct_pca':
        ltv = global_pen * tv_ratio
        ll1 = l1_ratio * global_pen * (1 - ltv)
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)

    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    n, p = X_train.shape
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]

    # A matrices
    Atv = GLOBAL.Atv
    Al1 = scipy.sparse.eye(p, p)

    # Fit model
    if model_name == 'pca':
        model = sklearn.decomposition.PCA(n_components=N_COMP)
    if model_name == 'sparse_pca':
        model = sklearn.decomposition.SparsePCA(n_components=N_COMP,
                                                alpha=ll1)
    if model_name == 'struct_pca':
        model = pca_tv.PCA_SmoothedL1_L2_TV(n_components=N_COMP,
                                            l1=ll1, l2=ll2, ltv=ltv,
                                            Atv=Atv,
                                            Al1=Al1,
                                            criterion="frobenius",
                                            eps=1e-6,
                                            max_iter=100,
                                            inner_max_iter=int(1e4),
                                            output=False)
    t0 = time.clock()
    model.fit(X_train)
    t1 = time.clock()
    _time = t1 - t0
    #print "X_test", GLOBAL.DATA["X"][1].shape

    # Save the projectors
    if (model_name == 'pca') or (model_name == 'sparse_pca'):
        V = model.components_.T
    if model_name == 'struct_pca':
        V = model.V

    # Project train & test data
    if (model_name == 'pca') or (model_name == 'sparse_pca'):
        X_train_transform = model.transform(X_train)
        X_test_transform = model.transform(X_test)
    if (model_name == 'struct_pca'):
        X_train_transform, _ = model.transform(X_train)
        X_test_transform, _ = model.transform(X_test)

    # Reconstruct train & test data
    # For SparsePCA or PCA, the formula is: UV^t (U is given by transform)
    # For StructPCA this is implemented in the predict method (which uses
    # transform)
    if (model_name == 'pca') or (model_name == 'sparse_pca'):
        X_train_predict = np.dot(X_train_transform, V.T)
        X_test_predict = np.dot(X_test_transform, V.T)
    if (model_name == 'struct_pca'):
        X_train_predict = model.predict(X_train)
        X_test_predict = model.predict(X_test)

    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')

    # Compute explained variance ratio
    evr_train = metrics.adjusted_explained_variance(X_train_transform)
    evr_train /= np.var(X_train, axis=0).sum()
    evr_test = metrics.adjusted_explained_variance(X_test_transform)
    evr_test /= np.var(X_test, axis=0).sum()

    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict

    # Compute geometric metrics and norms of components
    TV = parsimony.functions.nesterov.tv.TotalVariation(1, A=Atv)
    l0 = np.zeros((N_COMP,))
    l1 = np.zeros((N_COMP,))
    l2 = np.zeros((N_COMP,))
    tv = np.zeros((N_COMP,))
    for i in range(N_COMP):
        # Norms
        l0[i] = np.linalg.norm(V[:, i], 0)
        l1[i] = np.linalg.norm(V[:, i], 1)
        l2[i] = np.linalg.norm(V[:, i], 2)
        tv[i] = TV.f(V[:, i])

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform,
               evr_train=evr_train,
               evr_test=evr_test,
               l0=l0,
               l1=l1,
               l2=l2,
               tv=tv,
               time=_time)

    output_collector.collect(key, ret)


def reducer(key, values):
    global N_COMP
    import mapreduce as GLOBAL
    N_FOLDS = GLOBAL.N_FOLDS
    # N_FOLDS is the number of true folds (not the number of resamplings)
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.]
    # Avoid taking into account the fold 0
    if GLOBAL.FULL_RESAMPLE:
        values = [item.load() for item in values[1:]]
    else:
        values = [item.load() for item in values]

    # Load components: each file is n_voxelsxN_COMP matrix.
    # We stack them on the third dimension (folds)
    components = np.dstack([item["components"] for item in values])
    # Thesholded components (list of tuples (comp, threshold))
    thresh_components = np.empty(components.shape)
    thresholds = np.empty((N_COMP, N_FOLDS))
    for l in range(N_FOLDS):
        for k in range(N_COMP):
            thresh_comp, t = array_utils.arr_threshold_from_norm2_ratio(
                                components[:, k, l],
                                .99)
            thresh_components[:, k, l] = thresh_comp
            thresholds[k, l] = t
    frobenius_train = np.vstack([item["frobenius_train"] for item in values])
    frobenius_test = np.vstack([item["frobenius_test"] for item in values])
    l0 = np.vstack([item["l0"] for item in values])
    l1 = np.vstack([item["l1"] for item in values])
    l2 = np.vstack([item["l2"] for item in values])
    tv = np.vstack([item["tv"] for item in values])
    evr_train = np.vstack([item["evr_train"] for item in values])
    evr_test = np.vstack([item["evr_test"] for item in values])
    times = [item["time"] for item in values]

    # Average precision/recall across folds for each component
    av_frobenius_train = frobenius_train.mean(axis=0)
    av_frobenius_test = frobenius_test.mean(axis=0)
    av_evr_train = evr_train.mean(axis=0)
    av_evr_test = evr_test.mean(axis=0)
    av_l0 = l0.mean(axis=0)
    av_l1 = l1.mean(axis=0)
    av_l2 = l2.mean(axis=0)
    av_tv = tv.mean(axis=0)

    # Compute correlations of components between all folds
    n_corr = N_FOLDS * (N_FOLDS - 1) / 2
    correlations = np.zeros((N_COMP, n_corr))
    for k in range(N_COMP):
        R = np.corrcoef(np.abs(components[:, k, :].T))
        # Extract interesting coefficients (upper-triangle)
        correlations[k] = R[np.triu_indices_from(R, 1)]

    # Transform to z-score
    Z = 1. / 2. * np.log((1 + correlations) / (1 - correlations))
    # Average for each component
    z_bar = np.mean(Z, axis=1)
    # Transform back to average correlation for each component
    r_bar = (np.exp(2 * z_bar) - 1) / (np.exp(2 * z_bar) + 1)

    # Compute fleiss_kappa and DICE on thresholded components
    fleiss_kappas = np.empty(N_COMP)
    dice_bars = np.empty(N_COMP)
    for k in range(N_COMP):
        # One component accross folds
        thresh_comp = thresh_components[:, k, :]
        try:
            # Compute fleiss kappa statistics
            # The "raters" are the folds and we have 3 variables:
            #  - number of null coefficients
            #  - number of > 0 coefficients
            #  - number of < 0 coefficients
            # We build a (N_FOLDS, 3) table
            thresh_comp_signed = np.sign(thresh_comp)
            table = np.zeros((N_FOLDS, 3))
            table[:, 0] = np.sum(thresh_comp_signed == 0, 0)
            table[:, 1] = np.sum(thresh_comp_signed == 1, 0)
            table[:, 2] = np.sum(thresh_comp_signed == -1, 0)
            fleiss_kappa_stat = fleiss_kappa(table)
        except:
            fleiss_kappa_stat = 0.
        fleiss_kappas[k] = fleiss_kappa_stat
        try:
            # Paire-wise DICE coefficient (there is the same number than
            # pair-wise correlations)
            thresh_comp_n0 = thresh_comp != 0
            # Index of lines (folds) to use
            ij = [[i, j] for i in xrange(N_FOLDS)
                         for j in xrange(i + 1, N_FOLDS)]
            num = [np.sum(thresh_comp[idx[0], :] == thresh_comp[idx[1], :])
                   for idx in ij]
            denom = [(np.sum(thresh_comp_n0[idx[0], :]) + \
                      np.sum(thresh_comp_n0[idx[1], :]))
                     for idx in ij]
            dices = np.array([float(num[i]) / denom[i] for i in range(n_corr)])
            dice_bar = dices.mean()
        except:
            dice_bar = 0.
        dice_bars[k] = dice_bar

    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),
        ('frobenius_train', av_frobenius_train[0]),
        ('frobenius_test', av_frobenius_test[0]),
        ('correlation_0', r_bar[0]),
        ('correlation_1', r_bar[1]),
        ('correlation_2', r_bar[2]),
        ('correlation_mean', np.mean(r_bar)),
        ('kappa_0', fleiss_kappas[0]),
        ('kappa_1', fleiss_kappas[1]),
        ('kappa_2', fleiss_kappas[2]),
        ('kappa_mean', np.mean(fleiss_kappas)),
        ('dice_bar_0', dice_bars[0]),
        ('dice_bar_1', dice_bars[1]),
        ('dice_bar_2', dice_bars[2]),
        ('dice_bar_mean', np.mean(dice_bar)),
        ('evr_train_0', av_evr_train[0]),
        ('evr_train_1', av_evr_train[1]),
        ('evr_train_2', av_evr_train[2]),
        ('evr_test_0', av_evr_test[0]),
        ('evr_test_1', av_evr_test[1]),
        ('evr_test_2', av_evr_test[2]),
        ('l0_0', av_l0[0]),
        ('l0_1', av_l0[1]),
        ('l0_2', av_l0[2]),
        ('l1_0', av_l1[0]),
        ('l1_1', av_l1[1]),
        ('l1_2', av_l1[2]),
        ('l2_0', av_l2[0]),
        ('l2_1', av_l2[1]),
        ('l2_2', av_l2[2]),
        ('tv_0', av_tv[0]),
        ('tv_1', av_tv[1]),
        ('tv_2', av_tv[2]),
        ('time', np.mean(times))))

    return scores


def run_test(wd, config):
    print "In run_test"
    import mapreduce
    os.chdir(wd)
    params = config['params'][-1]
    key = '_'.join([str(p) for p in params])
    load_globals(config)
    OUTPUT = os.path.join('test', key)
    oc = mapreduce.OutputCollector(OUTPUT)
    X = np.load(config['data']['X'])
    mapreduce.DATA_RESAMPLED = {}
    mapreduce.DATA_RESAMPLED["X"] = [X, X]
    mapper(params, oc)


def create_config(y, n_folds, output_dir, filename,
                  include_full_resample=False):

    full_output_dir = os.path.join(OUTPUT_DIR, output_dir)
    if not os.path.exists(full_output_dir):
        os.makedirs(full_output_dir)

    skf = StratifiedKFold(y=y,
                          n_folds=n_folds)
    resample_index = [[tr.tolist(), te.tolist()] for tr, te in skf]
    if include_full_resample:
        resample_index.insert(0, None)  # first fold is None

    # Copy the learning data & mask
    shutil.copy2(INPUT_DATASET, full_output_dir)
    shutil.copy2(INPUT_MASK, full_output_dir)

    # Create config file
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.basename(INPUT_DATASET)),
                  mask_file=os.path.basename(INPUT_MASK),
                  params=PARAMS,
                  resample=resample_index,
                  map_output="results",
                  n_folds=n_folds,
                  full_resample=include_full_resample,
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results.csv")
    config_full_filename = os.path.join(full_output_dir, filename)
    json.dump(config, open(config_full_filename, "w"))

    # Create files to synchronize with the cluster
    sync_push_filename, sync_pull_filename, CLUSTER_WD = \
    clust_utils.gabriel_make_sync_data_files(full_output_dir)

    # Create job files
    cluster_cmd = "mapreduce.py -m {dir}/{file}  --ncore 12".format(
                            dir=CLUSTER_WD,
                            file=filename)
    clust_utils.gabriel_make_qsub_job_files(full_output_dir, cluster_cmd)

    return config

#################
# Actual script #
#################

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Read and map site (used to split groups)
    clinic_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
    clinic_subjects_id = clinic_data.index
    print "Found", len(clinic_subjects_id), "clinic records"
    y = clinic_data['SITE'].map({'FR': 0, 'GE': 1})

    # Create config files
    config_2folds = create_config(y, *(CONFIGS[0]))
    config_5folds = create_config(y, *(CONFIGS[1]))

    DEBUG = False
    if DEBUG:
        run_test(OUTPUT_DIR, config_2folds)
