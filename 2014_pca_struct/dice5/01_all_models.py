# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:08:39 2014

@author: md238665

Process dice5 datasets with standard PCA, sklearn SparsePCA and our
structured PCA.
We use several values for global penalization, TV ratio and L1 ratio.

We generate a map_reduce configuration file for each dataset and the files
needed to run on the cluster.
Due to the cluster setup and the way mapreduce work we need to copy the datsets
and the masks on the cluster. Therefore we copy them on the output directory
which is synchronised on the cluster.

The output directory is results/data_{shape}_{snr}.

"""

import os
import json
import time
import shutil

from itertools import product
from collections import OrderedDict

import numpy as np
import scipy

import sklearn.decomposition

import parsimony.functions.nesterov.tv
import pca_tv
import metrics

from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils

import dice5_metrics

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_BASE_DATA_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_DIR_FORMAT = os.path.join(INPUT_BASE_DATA_DIR,
                                "data_{s[0]}_{s[1]}_{snr}")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_INDEX_FILE_FORMAT = "indices_{subset}.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"

INPUT_SHAPE = (100, 100, 1)
INPUT_N_SUBSETS = 2
INPUT_SNRS = [0.1, 0.2, 0.25, 0.5, 1.0]

OUTPUT_BASE_DIR = os.path.join(INPUT_BASE_DIR, "results")
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,
                                 "data_{s[0]}_{s[1]}_{snr}")

##############
# Parameters #
##############

N_COMP = 3
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2])
# Relative penalties
# 0.33 ensures that there is a case with TV = L1 = L2
TVRATIO = np.array([1, 0.5, 0.33, 1e-1, 1e-2, 1e-3, 0])
L1RATIO = np.array([1, 0.5, 1e-1, 1e-2, 1e-3, 0])

PCA_PARAMS = [('pca', 0.0, 0.0, 0.0)]
SPARSE_PCA_PARAMS = list(product(['sparse_pca'],
                                 GLOBAL_PENALTIES,
                                 [0.0],
                                 [1.0]))
STRUCT_PCA_PARAMS = list(product(['struct_pca'],
                                 GLOBAL_PENALTIES,
                                 TVRATIO,
                                 L1RATIO))

PARAMS = PCA_PARAMS + SPARSE_PCA_PARAMS + STRUCT_PCA_PARAMS

#############
# Functions #
#############


def load_globals(config):
    global INPUT_OBJECT_MASK_FILE_FORMAT
    import mapreduce as GLOBAL
    # Read masks
    masks = []
    for i in range(N_COMP):
        filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
        masks.append(np.load(filename))
    im_shape = config["im_shape"]
    Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_shape(im_shape)
    GLOBAL.Atv = Atv
    GLOBAL.masks = masks


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...]
                                 for idx in resample]
                                 for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k]
                                 for idx in [0, 1]]
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
        ll1 = l1_ratio * global_pen * (1 - tv_ratio)
        ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        assert(np.allclose(ll1 + ll2 + ltv, global_pen))

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

    # Compute geometric metrics and norms of components
    TV = parsimony.functions.nesterov.tv.TotalVariation(1, A=Atv)
    l0 = np.zeros((N_COMP,))
    l1 = np.zeros((N_COMP,))
    l2 = np.zeros((N_COMP,))
    tv = np.zeros((N_COMP,))
    recall = np.zeros((N_COMP,))
    precision = np.zeros((N_COMP,))
    fscore = np.zeros((N_COMP,))
    for i in range(N_COMP):
        # Norms
        l0[i] = np.linalg.norm(V[:, i], 0)
        l1[i] = np.linalg.norm(V[:, i], 1)
        l2[i] = np.linalg.norm(V[:, i], 2)
        tv[i] = TV.f(V[:, i])

    # Compute explained variance ratio
    evr_train = metrics.adjusted_explained_variance(X_train_transform)
    evr_train /= np.var(X_train, axis=0).sum()
    evr_test = metrics.adjusted_explained_variance(X_test_transform)
    evr_test /= np.var(X_test, axis=0).sum()

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=V,
               X_train_transform=X_train_transform,
               X_test_transform=X_test_transform,
               X_train_predict=X_train_predict,
               X_test_predict=X_test_predict,
               recall=recall,
               precision=precision,
               fscore=fscore,
               evr_train=evr_train,
               evr_test=evr_test,
               l0=l0,
               l1=l1,
               l2=l2,
               tv=tv,
               time=_time)

    output_collector.collect(key, ret)


def reducer(key, values):
    output_collectors = values
    global N_COMP, INPUT_N_SUBSETS
    import mapreduce as GLOBAL
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.]
    # Avoid taking into account the fold 0
    values = [item.load() for item in output_collectors[1:]]

    N_FOLDS = INPUT_N_SUBSETS
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
    # Save thresholded comp
    for l, oc in zip(range(N_FOLDS), output_collectors[1:]):
        filename = os.path.join(oc.output_dir, "thresh_comp.npz")
        np.savez(filename, thresh_components[:, :, l])
    frobenius_train = np.vstack([item["frobenius_train"] for item in values])
    frobenius_test = np.vstack([item["frobenius_test"] for item in values])
    l0 = np.vstack([item["l0"] for item in values])
    l1 = np.vstack([item["l1"] for item in values])
    l2 = np.vstack([item["l2"] for item in values])
    tv = np.vstack([item["tv"] for item in values])
    evr_train = np.vstack([item["evr_train"] for item in values])
    evr_test = np.vstack([item["evr_test"] for item in values])
    times = [item["time"] for item in values]

    # Compute precision/recall for each component and fold
    precisions = np.zeros((N_COMP, N_FOLDS))
    recalls = np.zeros((N_COMP, N_FOLDS))
    fscores = np.zeros((N_COMP, N_FOLDS))
    for k in range(N_COMP):
        for n in range(N_FOLDS):
            c = components[:, k, n]
            precisions[k, n], recalls[k, n], fscores[k, n] = \
              dice5_metrics.dice_five_geometric_metrics(GLOBAL.masks[k], c)

    # Compute precision/recall for each thresholded component and fold
    thresh_precisions = np.zeros((N_COMP, N_FOLDS))
    thresh_recalls = np.zeros((N_COMP, N_FOLDS))
    thresh_fscores = np.zeros((N_COMP, N_FOLDS))
    for k in range(N_COMP):
        for n in range(N_FOLDS):
            c = thresh_components[:, k, n]
            thresh_precisions[k, n], thresh_recalls[k, n], thresh_fscores[k, n] = \
              dice5_metrics.dice_five_geometric_metrics(GLOBAL.masks[k], c)

    # Average precision/recall across folds for each component
    av_frobenius_train = frobenius_train.mean(axis=0)
    av_frobenius_test = frobenius_test.mean(axis=0)
    av_precision = precisions.mean(axis=1)
    av_recall = recalls.mean(axis=1)
    av_fscore = fscores.mean(axis=1)
    av_thresh_precision = thresh_precisions.mean(axis=1)
    av_thresh_recall = thresh_recalls.mean(axis=1)
    av_thresh_fscore = thresh_fscores.mean(axis=1)
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

    # Align sign of loading vectors to the first fold for each component
    aligned_thresh_comp = np.copy(thresh_components)
    REF_FOLD_NUMBER = 0
    for k in range(N_COMP):
        for i in range(N_FOLDS):
            ref = thresh_components[:, k, REF_FOLD_NUMBER].T
            if i != REF_FOLD_NUMBER:
                r = np.corrcoef(thresh_components[:, k, i].T,
                                ref)
                if r[0, 1] < 0:
                    #print "Reverting comp {k} of fold {i} for model {key}".format(i=i+1, k=k, key=key)
                    aligned_thresh_comp[:, k, i] *= -1
    # Save aligned comp
    for l, oc in zip(range(N_FOLDS), output_collectors[1:]):
        filename = os.path.join(oc.output_dir, "aligned_thresh_comp.npz")
        np.savez(filename, aligned_thresh_comp[:, :, l])

    # Compute fleiss_kappa and DICE on thresholded components
    fleiss_kappas = np.empty(N_COMP)
    dice_bars = np.empty(N_COMP)
    for k in range(N_COMP):
        # One component accross folds
        thresh_comp = aligned_thresh_comp[:, k, :]
        fleiss_kappas[k] = metrics.fleiss_kappa(thresh_comp)
        dice_bars[k] = metrics.dice_bar(thresh_comp)

    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),

        ('frobenius_train', av_frobenius_train[0]),
        ('frobenius_test', av_frobenius_test[0]),

        ('recall_0', av_recall[0]),
        ('recall_1', av_recall[1]),
        ('recall_2', av_recall[2]),
        ('recall_mean', np.mean(av_recall)),
        ('precision_0', av_precision[0]),
        ('precision_1', av_precision[1]),
        ('precision_2', av_precision[2]),
        ('precision_mean', np.mean(av_precision)),
        ('fscore_0', av_fscore[0]),
        ('fscore_1', av_fscore[1]),
        ('fscore_2', av_fscore[2]),
        ('fscore_mean', np.mean(av_fscore)),

        ('thresh_recall_0', av_thresh_recall[0]),
        ('thresh_recall_1', av_thresh_recall[1]),
        ('thresh_recall_2', av_thresh_recall[2]),
        ('thresh_recall_mean', np.mean(av_thresh_recall)),
        ('thresh_precision_0', av_thresh_precision[0]),
        ('thresh_precision_1', av_thresh_precision[1]),
        ('thresh_precision_2', av_thresh_precision[2]),
        ('thresh_precision_mean', np.mean(av_thresh_precision)),
        ('thresh_fscore_0', av_thresh_fscore[0]),
        ('thresh_fscore_1', av_thresh_fscore[1]),
        ('thresh_fscore_2', av_thresh_fscore[2]),
        ('thresh_fscore_mean', np.mean(av_thresh_fscore)),

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
        ('dice_bar_mean', np.mean(dice_bars)),
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

#################
# Actual script #
#################

if __name__ == '__main__':
    # Create a mapreduce config file for each dataset
    for snr in INPUT_SNRS:
        input_dir = INPUT_DIR_FORMAT.format(s=INPUT_SHAPE,
                                            snr=snr)
        # Read indices
        indices = []
        for i in range(INPUT_N_SUBSETS):
            filename = INPUT_INDEX_FILE_FORMAT.format(subset=i)
            full_filename = os.path.join(input_dir, filename)
            indices.append(np.load(full_filename).tolist())
        rev_indices = indices[::-1]
        resample_index = [indices, rev_indices]
        resample_index.insert(0, None)  # first fold is None

        # Local output directory for this dataset
        output_dir = os.path.join(OUTPUT_BASE_DIR,
                                  OUTPUT_DIR_FORMAT.format(s=INPUT_SHAPE,
                                                           snr=snr))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Copy the learning data
        src_datafile = os.path.join(input_dir, INPUT_STD_DATASET_FILE)
        shutil.copy(src_datafile, output_dir)

        # Copy the objects masks
        for i in range(N_COMP):
            filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
            src_filename = os.path.join(input_dir, filename)
            dst_filename = os.path.join(output_dir, filename)
            shutil.copy(src_filename, dst_filename)

        # Create files to synchronize with the cluster
        sync_push_filename, sync_pull_filename, CLUSTER_WD = \
        clust_utils.gabriel_make_sync_data_files(output_dir)

        # Create config file
        user_func_filename = os.path.abspath(__file__)

        config = dict(data=dict(X=INPUT_STD_DATASET_FILE),
                      im_shape=INPUT_SHAPE,
                      params=PARAMS,
                      resample=resample_index,
                      map_output="results",
                      user_func=user_func_filename,
                      ncore=4,
                      reduce_group_by="params",
                      reduce_output="results.csv")
        config_full_filename = os.path.join(output_dir, "config.json")
        json.dump(config, open(config_full_filename, "w"))

        # Create job files
        cluster_cmd = "mapreduce.py -m %s/config.json  --ncore 12" % CLUSTER_WD
        clust_utils.gabriel_make_qsub_job_files(output_dir, cluster_cmd)

    DEBUG = False
    if DEBUG:
        run_test(output_dir, config)
