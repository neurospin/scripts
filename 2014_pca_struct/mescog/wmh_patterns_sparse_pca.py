# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:45 2014

@author: md238665

Read the data generated in 2013_mescog/proj_wmh_patterns.

We use the centered data

"""

import os
import json

import numpy as np

import sklearn
import sklearn.decomposition
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd

import nibabel

import matplotlib.pyplot as plt

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns")

# 
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "all.centered.npy")

INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/brainomics"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "2014_pca_struct", "mescog")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Output for scikit-learn sparse PCA: alpha will be replaced by actual value
OUTPUT_DIR  = os.path.join(OUTPUT_DIR, "SparsePCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
#OUTPUT_SPARSE_PCA_PRED = os.path.join(OUTPUT_SPARSE_PCA_DIR, "X_pred.npy")
#OUTPUT_SPARSE_PCA_COMP = os.path.join(OUTPUT_SPARSE_PCA_DIR, "components.npy")

##############
# Parameters #
##############

N_COMP = 5
SPARSE_PCA_ALPHA = np.arange(0, 10, 1)

#############
# Functions #
#############

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    # GLOBAL.DATA
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, = key
    sparse_pca = sklearn.decomposition.SparsePCA(n_components=10,
                                                 alpha=alpha)
    sparse_pca.fit(GLOBAL.DATA["X"][0])
    X_transform = sparse_pca.transform(GLOBAL.DATA["X"][1])
    ret = dict(X_transform=X_transform,
               V=sparse_pca.V,
               U=sparse_pca.U)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    return None


#################
# Actual script #
#################

# Read learning data (french & german subjects)
#X = np.load(INPUT_DATASET)
#print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

## Create config file
# parameters grid
params = [SPARSE_PCA_ALPHA.tolist(), ]
# User map/reduce function file:
user_func_filename = os.path.abspath(__file__)

config = dict(data=dict(X=INPUT_DATASET),
              params=params,
              map_output=os.path.join(OUTPUT_DIR, "results"),
              user_func=user_func_filename,
              ncore=4,
              reduce_input=os.path.join(OUTPUT_DIR, "results/*/*"),
              reduce_group_by=os.path.join(OUTPUT_DIR, "results/.*/(.*)"),
              reduce_output=os.path.join(OUTPUT_DIR, "results.csv"))
json.dump(config, open(os.path.join(OUTPUT_DIR, "config.json"), "w"))

#############################################################################
print "# Start by running Locally with 2 cores, to check that everything is OK)"
print "Interrupt after a while CTL-C"
print "mapreduce.py --mode map --config %s/config.json --ncore 2" % OUTPUT_DIR
#os.system("mapreduce.py --mode map --config %s/config.json" % WD)
#
##############################################################################
#    print "# Run on the cluster with 30 PBS Jobs"
#    print "mapreduce.py --pbs_njob 30 --config %s/config.json" % OUTPUT
#    
#    #############################################################################
#    print "# Reduce"
#    print "mapreduce.py --mode reduce --config %s/config.json" % OUTPUT
#
## Sparse PCA
#for alpha in SPARSE_PCA_ALPHA:
#    sparse_pca = sklearn.decomposition.SparsePCA(n_components=10,
#                                                alpha=alpha)
#    X_sparse_pca = sparse_pca.fit(X).transform(X)
#    output_dir = OUTPUT_SPARSE_PCA_DIR.format(alpha=alpha)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    np.save(OUTPUT_SPARSE_PCA_PRED.format(alpha=alpha), X_sparse_pca)
#    np.save(OUTPUT_SPARSE_PCA_COMP.format(alpha=alpha), sparse_pca.components_)
