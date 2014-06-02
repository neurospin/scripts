# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:08:39 2014

@author: md238665

Process dice5 datasets with our structured PCA.
We use very few TV here.

We generate a map_reduce configuration file for each dataset.

"""

import os
import json
import time

from collections import OrderedDict

import numpy as np
import scipy

import parsimony.functions.nesterov.tv
import pca_tv

import dice5_pca

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_STD_DATASET_FILE_FORMAT = "data_{alpha}.std.npy"
INPUT_INDEX_FILE_FORMAT = "{subset}_indices.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"

INPUT_SHAPE = (100, 100, 1)
INPUT_ALPHAS = np.linspace(0.1, 1, num=10)
INPUT_TRAIN_INDICES = os.path.join(INPUT_DIR, "train_indices.npy")
INPUT_TEST_INDICES = os.path.join(INPUT_DIR, "test_indices.npy")
INPUT_BETA_FILE_FORMAT = "beta3d_{alpha}.std.npy"

OUTPUT_DIR = os.path.join(INPUT_BASE_DIR, "struct_pca")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_DATASET_DIR = "{alpha}"

##############
# Parameters #
##############

N_COMP = 3
# Global penalty
STRUCTPCA_ALPHA = np.array([1, 5, 10])
# Relative penalties
STRUCTPCA_L1L2TV = np.array([0.1, 1e-3, 1e-6])

#############
# Functions #
#############

#def A_from_structure(structure_filepath):
#

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    # GLOBAL.DATA
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    ll1, ll2, ltv = key
    # A matrices
    X_train = GLOBAL.DATA["X"][0]
    n, p = shape = X_train.shape
    #print "X_train", shape
    Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_shape(INPUT_SHAPE)
    Al1 = scipy.sparse.eye(p, p)
    model = pca_tv.PCA_SmoothedL1_L2_TV(n_components=N_COMP,
                                        l1=ll1, l2=ll2, ltv=ltv,
                                        Atv=Atv,
                                        Al1=Al1,
                                        criterion="frobenius",
                                        eps=1e-6,
                                        inner_max_iter=int(1e5),
                                        output=False)
    t0 = time.clock()
    model.fit(X_train)
    t1 = time.clock()
    #print "X_test", GLOBAL.DATA["X"][1].shape
    X_transform = model.transform(GLOBAL.DATA["X"][1])
    ret = dict(X_train=X_train,
               X_transform=X_transform,
               model=model,
               time=t1-t0)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # load return dict corresponding to mapper ouput. they need to be loaded.

    values = [item.load() for item in values]
    model = values[0]["model"]

    # Read masks & compute geometric metrics
    V = model.V
    global INPUT_OBJECT_MASK_FILE_FORMAT, INPUT_BETA_FILE_FORMAT
    recall = []
    precision = []
    corr = []
    for i in range(3):
        filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
        full_filename = os.path.join(INPUT_DIR, filename)
        print full_filename
        mask = np.load(full_filename)
        #masks.append(mask)
        res = dice5_pca.dice_five_geometric_metrics(mask, V[:, i])
        recall.append(res[0])
        precision.append(res[1])
        corr.append(res[2])

    # Percentage of explained variance
    X_train = values[0]["X_train"]
    d2 = model.d**2
    variance = np.sum(X_train**2)
    evr2 = d2.cumsum()/variance

    evr  = []
    for i in range(1,4):
        evr.append(dice5_pca.ratio_explained_variance(X_train, V[:, 0:i]))

    time = values[0]["time"]
    scores = dict(key=key,
                  recall_0=recall[0], recall_1=recall[1],
                  recall_2=recall[2], recall_mean=np.mean(recall),
                  precision_0=precision[0], precision_1=precision[1],
                  precision_2=precision[2], precision_mean=np.mean(precision),
                  corr_0=corr[0], corr_1=corr[1],
                  corr_2=corr[2], corr_mean=np.mean(corr),
                  evr_witten_0=evr[0], evr_witten_1=evr[1],
                  evr_witten_2=evr[2],
                  evr_0=evr2[0],evr_1=evr2[1],
                  evr_2=evr2[2],
                  time=time)
    return OrderedDict(sorted(scores.items(), key=lambda t: t[0]))

#################
# Actual script #
#################

if __name__ == '__main__':

    # Read indices
    train_indices = np.load(INPUT_TRAIN_INDICES).tolist()
    test_indices = np.load(INPUT_TEST_INDICES).tolist()

    # Parameter grid
    params = [(alpha * STRUCTPCA_L1L2TV).tolist() for alpha in STRUCTPCA_ALPHA]

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
                      params=params,
                      resample=[[train_indices, test_indices]],
                      map_output=output_dir,
                      user_func=user_func_filename,
                      ncore=4,
                      reduce_input=os.path.join(output_dir, "*/*"),
                      reduce_group_by=os.path.join(output_dir, ".*/(.*)"),
                      reduce_output=os.path.join(output_dir, "results.csv"))
        config_full_filename = os.path.join(output_dir, "config.json")
        json.dump(config, open(config_full_filename, "w"))
        print "mapreduce.py --mode map --config %s" % config_full_filename
