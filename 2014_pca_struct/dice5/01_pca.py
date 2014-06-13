# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:08:39 2014

@author: md238665

Process dice5 datasets with standard PCA.

We generate a map_reduce configuration file for each dataset.
In this case this a bit an overkill since there is no parameters.

"""

import os
import json
import time

from collections import OrderedDict

import numpy as np
import scipy

import sklearn.decomposition

import dice5_pca

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_STD_DATASET_FILE_FORMAT = "data_{alpha}.std.npy"
INPUT_INDEX_FILE_FORMAT = "indices_{subset}.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"

INPUT_SHAPE = (100, 100, 1)
INPUT_N_SUBSETS = 2
INPUT_ALPHAS = np.linspace(0.1, 1, num=10)
INPUT_BETA_FILE_FORMAT = "beta3d_{alpha}.std.npy"

OUTPUT_DIR = os.path.join(INPUT_BASE_DIR, "pca")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_DATASET_DIR = "{alpha}"

##############
# Parameters #
##############

N_COMP = 3

#############
# Functions #
#############

def load_globals(config):
    pass


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    global INPUT_OBJECT_MASK_FILE_FORMAT, INPUT_BETA_FILE_FORMAT, N_COMP
    import mapreduce as GLOBAL # access to global variables:
    # GLOBAL.DATA
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    X_train = GLOBAL.DATA_RESAMPLED["X"][0]

    model = sklearn.decomposition.PCA(n_components=N_COMP)
    t0 = time.clock()
    model.fit(X_train)
    t1 = time.clock()
    _time = t1-t0
    #print "X_test", GLOBAL.DATA["X"][1].shape

    # Transform data
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
    X_transform = model.transform(X_test)

    # Read masks & compute geometric metrics
    V = model.components_.T
    recall = np.zeros((N_COMP,))
    precision = np.zeros((N_COMP,))
    fscore = np.zeros((N_COMP,))
    for i in range(N_COMP):
        filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
        full_filename = os.path.join(INPUT_DIR, filename)
        #print full_filename
        mask = np.load(full_filename)
        #masks.append(mask)
        res = dice5_pca.dice_five_geometric_metrics(mask, V[:, i])
        recall[i] = res[0]
        precision[i] = res[1]
        fscore[i] = res[2]

    evr_shen = np.zeros((N_COMP,))
    evr_zou = np.zeros((N_COMP,))
    for i in range(N_COMP):
        # i first components
        Vi = V[:, range(i+1)]
        try:
            evr_shen[i] = dice5_pca.explained_variance_shen(X_train, Vi)
        except:
            evr_shen[i] = np.nan
        try:
            evr_zou[i] = dice5_pca.explained_variance_zou(X_train, Vi)
        except:
            evr_zou[i] = np.nan

    ret = dict(X_transform=X_transform,
               model=model,
               recall=recall,
               precision=precision,
               fscore=fscore,
               evr_shen=evr_shen,
               evr_zou=evr_zou,
               time=_time)

    output_collector.collect(key, ret)


def reducer(key, values):
    global N_COMP
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.]
    values = [item.load() for item in values]

    models = [item["model"] for item in values]
    precisions = np.vstack([item["precision"] for item in values])
    recalls = np.vstack([item["recall"] for item in values])
    fscores = np.vstack([item["fscore"] for item in values])
    evr_shen = np.vstack([item["evr_shen"] for item in values])
    evr_zou = np.vstack([item["evr_zou"] for item in values])
    times = [item["time"] for item in values]

    # Average precision/recall across folds for each group
    av_precision = precisions.mean(axis=0)
    av_recall = recalls.mean(axis=0)
    av_fscore = fscores.mean(axis=0)
    av_evr_shen = evr_shen.mean(axis=0)
    av_evr_zou = evr_zou.mean(axis=0)

    # Compute correlations of components between folds
    correlation = np.zeros((N_COMP,))
    comp0 = models[0].components_.T
    comp1 = models[1].components_.T
    for i in range(N_COMP):
        correlation[i] = dice5_pca.abs_correlation(comp0[:, i], comp1[:, i])

    scores = dict(key=key,
                  recall_0=av_recall[0], recall_1=av_recall[1],
                  recall_2=av_recall[2], recall_mean=np.mean(av_recall),
                  precision_0=av_precision[0], precision_1=av_precision[1],
                  precision_2=av_precision[2], precision_mean=np.mean(av_precision),
                  fscore_0=av_fscore[0], fscore_1=av_fscore[1],
                  fscore_2=av_fscore[2], fscore_mean=np.mean(av_fscore),
                  correlation_0=correlation[0], correlation_1=correlation[1],
                  correlation_2=correlation[2], correlation_mean=np.mean(correlation),
                  evr_shen_0=av_evr_shen[0], evr_1=av_evr_shen[1],
                  evr_shen_2=av_evr_shen[2],
                  evr_zou_0=av_evr_zou[0], evr_zou_1=av_evr_zou[1],
                  evr_zou_2=av_evr_zou[2],
                  time=np.mean(times))

    return OrderedDict(sorted(scores.items(), key=lambda t: t[0]))

#################
# Actual script #
#################

if __name__ == '__main__':

    # Read indices
    indices = []
    for i in range(INPUT_N_SUBSETS):
        filename = INPUT_INDEX_FILE_FORMAT.format(subset=i)
        full_filename = os.path.join(INPUT_DIR, filename)
        indices.append(np.load(full_filename).tolist())
    rev_indices = indices[::-1]

    # Create a mapreduce config file for each dataset
    for alpha in INPUT_ALPHAS:
        # Read learning data
        filename = INPUT_STD_DATASET_FILE_FORMAT.format(alpha=alpha)
        dataset_full_filename = os.path.join(INPUT_DIR, filename)

        # Output directory for this dataset
        output_dir = os.path.join(OUTPUT_DIR, OUTPUT_DATASET_DIR.format(alpha=alpha))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create config file
        # User map/reduce function file
        user_func_filename = os.path.abspath(__file__)

        config = dict(data=dict(X=dataset_full_filename),
                      params=[['PCA']],
                      resample=[indices,
                                rev_indices],
                      map_output=output_dir,
                      user_func=user_func_filename,
                      ncore=4,
                      reduce_input=os.path.join(output_dir, "*/*"),
                      reduce_group_by=os.path.join(output_dir, ".*/(.*)"),
                      reduce_output=os.path.join(output_dir, "results.csv"))
        config_full_filename = os.path.join(output_dir, "config.json")
        json.dump(config, open(config_full_filename, "w"))
        print "mapreduce.py -m %s" % config_full_filename
