# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:08:39 2014

@author: md238665

Process dice5 datasets with standard PCA and our structured PCA.
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

import dice5_data
import dice5_metrics

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_BASE_DATA_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_MASK_DIR = os.path.join(INPUT_BASE_DATA_DIR, "masks")
INPUT_DATA_DIR_FORMAT = os.path.join(INPUT_BASE_DATA_DIR,
                                     "data_{s[0]}_{s[1]}_{snr}")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"
INPUT_L1MASK_FILE = "l1_max.txt"
INPUT_SNR_FILE = os.path.join(INPUT_BASE_DIR,
                              "calibrate",
                              "SNR.npy")

OUTPUT_BASE_DIR = os.path.join(INPUT_BASE_DIR, "results")
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,
                                 "data_{s[0]}_{s[1]}_{snr}")

##############
# Parameters #
##############

N_COMP = 3
TRAIN_RANGE = range(dice5_data.N_SAMPLES/2)
TEST_RANGE = range(dice5_data.N_SAMPLES/2, dice5_data.N_SAMPLES)
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1, 1e1])
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

JSON_DUMP_OPT = {'indent': 4,
                 'separators': (',', ': ')}

#############
# Functions #
#############


def compute_coefs_from_ratios(global_pen, tv_ratio, l1_ratio):
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
    assert(np.allclose(ll1 + ll2 + ltv, global_pen))
    return ll1, ll2, ltv


def load_globals(config):
    global INPUT_OBJECT_MASK_FILE_FORMAT, N_COMP
    import mapreduce as GLOBAL
    # Read masks
    masks = []
    for i in range(N_COMP):
        filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
        masks.append(np.load(filename))
    im_shape = config["im_shape"]
    Atv = parsimony.functions.nesterov.tv.A_from_shape(im_shape)
    GLOBAL.Atv = Atv
    GLOBAL.masks = masks
    GLOBAL.l1_max = config["l1_max"]
    GLOBAL.N_COMP = config["n_comp"]


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
    if model_name == 'struct_pca':
        ll1, ll2, ltv = compute_coefs_from_ratios(global_pen,
                                                  tv_ratio,
                                                  l1_ratio)
        # This should not happen
        if ll1 > GLOBAL.l1_max:
            raise ValueError

    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    n, p = X_train.shape
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]

    # A matrices
    Atv = GLOBAL.Atv

    # Fit model
    if model_name == 'pca':
        model = sklearn.decomposition.PCA(n_components=GLOBAL.N_COMP)
    if model_name == 'struct_pca':
        model = pca_tv.PCA_L1_L2_TV(n_components=GLOBAL.N_COMP,
                                    l1=ll1, l2=ll2, ltv=ltv,
                                    Atv=Atv,
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
    if (model_name == 'pca'):
        components = model.components_.T
    if model_name == 'struct_pca':
        components = model.V

    # Threshold components
    thresh_components = np.empty(components.shape)
    thresholds = np.empty((GLOBAL.N_COMP, ))
    for k in range(GLOBAL.N_COMP):
        thresh_comp, t = array_utils.arr_threshold_from_norm2_ratio(
            components[:, k],
            .99)
        thresh_components[:, k] = thresh_comp
        thresholds[k] = t

    # Project train & test data
    if (model_name == 'pca'):
        X_train_transform = model.transform(X_train)
        X_test_transform = model.transform(X_test)
    if (model_name == 'struct_pca'):
        X_train_transform, _ = model.transform(X_train)
        X_test_transform, _ = model.transform(X_test)

    # Reconstruct train & test data
    # For PCA, the formula is: UV^t (U is given by transform)
    # For StructPCA this is implemented in the predict method (which uses
    # transform)
    if (model_name == 'pca'):
        X_train_predict = np.dot(X_train_transform, components.T)
        X_test_predict = np.dot(X_test_transform, components.T)
    if (model_name == 'struct_pca'):
        X_train_predict = model.predict(X_train)
        X_test_predict = model.predict(X_test)

    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')

    # Compute geometric metrics and norms of components
    TV = parsimony.functions.nesterov.tv.TotalVariation(1, A=Atv)
    l0 = np.zeros((GLOBAL.N_COMP,))
    l1 = np.zeros((GLOBAL.N_COMP,))
    l2 = np.zeros((GLOBAL.N_COMP,))
    tv = np.zeros((GLOBAL.N_COMP,))
    recall = np.zeros((GLOBAL.N_COMP,))
    precision = np.zeros((GLOBAL.N_COMP,))
    fscore = np.zeros((GLOBAL.N_COMP,))
    for i in range(GLOBAL.N_COMP):
        # Norms
        l0[i] = np.linalg.norm(components[:, i], 0)
        l1[i] = np.linalg.norm(components[:, i], 1)
        l2[i] = np.linalg.norm(components[:, i], 2)
        tv[i] = TV.f(components[:, i])

    # Compute explained variance ratio
    evr_train = metrics.adjusted_explained_variance(X_train_transform)
    evr_train /= np.var(X_train, axis=0).sum()
    evr_test = metrics.adjusted_explained_variance(X_test_transform)
    evr_test /= np.var(X_test, axis=0).sum()

    ret = dict(frobenius_train=frobenius_train,
               frobenius_test=frobenius_test,
               components=components,
               thresh_components=thresh_components,
               thresholds=thresholds,
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
    import mapreduce as GLOBAL
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.]
    if len(output_collectors) > 1:
        raise ValueError
    values = output_collectors[0].load()

    # Load component (n_voxelsxN_COMP matrix).
    components = values["components"]
    thresh_components = values["thresh_components"]

    frobenius_train = values["frobenius_train"]
    frobenius_test = values["frobenius_test"]
    l0 = values["l0"]
    l1 = values["l1"]
    l2 = values["l2"]
    tv = values["tv"]
    evr_train = values["evr_train"]
    evr_test = values["evr_test"]
    time = values["time"]

    # Compute precision/recall for each component
    precisions = np.zeros((GLOBAL.N_COMP, ))
    recalls = np.zeros((GLOBAL.N_COMP, ))
    fscores = np.zeros((GLOBAL.N_COMP, ))
    for k in range(GLOBAL.N_COMP):
        c = components[:, k]
        precisions[k], recalls[k], fscores[k] = \
            dice5_metrics.geometric_metrics(GLOBAL.masks[k], c)

    # Compute precision/recall for each thresholded component
    thresh_precisions = np.zeros((GLOBAL.N_COMP, ))
    thresh_recalls = np.zeros((GLOBAL.N_COMP, ))
    thresh_fscores = np.zeros((GLOBAL.N_COMP, ))
    for k in range(GLOBAL.N_COMP):
        c = thresh_components[:, k]
        thresh_precisions[k], thresh_recalls[k], thresh_fscores[k] = \
            dice5_metrics.geometric_metrics(GLOBAL.masks[k], c)

    scores = OrderedDict((
        ('model', key[0]),
        ('global_pen', key[1]),
        ('tv_ratio', key[2]),
        ('l1_ratio', key[3]),

        ('frobenius_train', frobenius_train),
        ('frobenius_test', frobenius_test),

        ('recall_0', recalls[0]),
        ('recall_1', recalls[1]),
        ('recall_2', recalls[2]),
        ('recall_mean', np.mean(recalls)),
        ('precision_0', precisions[0]),
        ('precision_1', precisions[1]),
        ('precision_2', precisions[2]),
        ('precision_mean', np.mean(precisions)),
        ('fscore_0', fscores[0]),
        ('fscore_1', fscores[1]),
        ('fscore_2', fscores[2]),
        ('fscore_mean', np.mean(fscores)),

        ('thresh_recall_0', thresh_recalls[0]),
        ('thresh_recall_1', thresh_recalls[1]),
        ('thresh_recall_2', thresh_recalls[2]),
        ('thresh_recall_mean', np.mean(thresh_recalls)),
        ('thresh_precision_0', thresh_precisions[0]),
        ('thresh_precision_1', thresh_precisions[1]),
        ('thresh_precision_2', thresh_precisions[2]),
        ('thresh_precision_mean', np.mean(thresh_precisions)),
        ('thresh_fscore_0', thresh_fscores[0]),
        ('thresh_fscore_1', thresh_fscores[1]),
        ('thresh_fscore_2', thresh_fscores[2]),
        ('thresh_fscore_mean', np.mean(thresh_fscores)),

        ('evr_train_0', evr_train[0]),
        ('evr_train_1', evr_train[1]),
        ('evr_train_2', evr_train[2]),
        ('evr_test_0', evr_test[0]),
        ('evr_test_1', evr_test[1]),
        ('evr_test_2', evr_test[2]),
        ('l0_0', l0[0]),
        ('l0_1', l0[1]),
        ('l0_2', l0[2]),
        ('l1_0', l1[0]),
        ('l1_1', l1[1]),
        ('l1_2', l1[2]),
        ('l2_0', l2[0]),
        ('l2_1', l2[1]),
        ('l2_2', l2[2]),
        ('tv_0', tv[0]),
        ('tv_1', tv[1]),
        ('tv_2', tv[2]),
        ('time', time)
    ))

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
    # Read SNRs
    input_snrs = np.load(INPUT_SNR_FILE)
    # Resample
    resamplings = [[TRAIN_RANGE, TEST_RANGE]]
    # Create a mapreduce config file for each dataset
    for snr in input_snrs:
        input_dir = INPUT_DATA_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                 snr=snr)
        # Read l1_max file
        full_filename = os.path.join(input_dir, INPUT_L1MASK_FILE)
        with open(full_filename) as f:
            S = f.readline()
            l1_max = float(S)

        # Remove configurations for which l1 > l1_max
        correct_params = []
        for params in PARAMS:
            model_name, global_pen, tv_ratio, l1_ratio = params
            if model_name != 'struct_pca':
                correct_params.append(params)
            else:
                ll1, _, _ = compute_coefs_from_ratios(global_pen,
                                                      tv_ratio,
                                                      l1_ratio)
                if ll1 < l1_max:
                    correct_params.append(params)

        # Local output directory for this dataset
        output_dir = os.path.join(OUTPUT_BASE_DIR,
                                  OUTPUT_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                           snr=snr))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Copy the learning data
        src_datafile = os.path.join(input_dir, INPUT_STD_DATASET_FILE)
        shutil.copy(src_datafile, output_dir)

        # Copy the objects masks
        for i in range(N_COMP):
            filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
            src_filename = os.path.join(INPUT_MASK_DIR, filename)
            dst_filename = os.path.join(output_dir, filename)
            shutil.copy(src_filename, dst_filename)

        # Create files to synchronize with the cluster
        sync_push_filename, sync_pull_filename, CLUSTER_WD = \
            clust_utils.gabriel_make_sync_data_files(output_dir,
                                                     user="md238665")

        # Create config file
        user_func_filename = os.path.abspath(__file__)

        config = OrderedDict([
            ('data', dict(X=INPUT_STD_DATASET_FILE)),
            ('im_shape', dice5_data.SHAPE),
            ('params', correct_params),
            ('l1_max', l1_max),
            ('n_comp', N_COMP),
            ('resample', resamplings),
            ('map_output', "results"),
            ('user_func', user_func_filename),
            ('ncore', 4),
            ('reduce_group_by', "params"),
            ('reduce_output', "results.csv")])
        config_full_filename = os.path.join(output_dir, "config.json")
        json.dump(config,
                  open(config_full_filename, "w"),
                  **JSON_DUMP_OPT)

        # Create job files
        cluster_cmd = "mapreduce.py -m %s/config.json  --ncore 12" % CLUSTER_WD
        clust_utils.gabriel_make_qsub_job_files(output_dir, cluster_cmd)

    DEBUG = False
    if DEBUG:
        run_test(output_dir, config)
