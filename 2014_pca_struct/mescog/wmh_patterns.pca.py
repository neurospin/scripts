# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:45 2014

@author: md238665

Read the data generated in 2013_mescog/proj_wmh_patterns.

We use the centered data

"""

import os

import numpy as np

import sklearn
import sklearn.decomposition

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

# Output for standard PCA
OUTPUT_PCA_DIR  = os.path.join(OUTPUT_DIR, "PCA")
if not os.path.exists(OUTPUT_PCA_DIR):
    os.makedirs(OUTPUT_PCA_DIR)
OUTPUT_PCA_PRED = os.path.join(OUTPUT_PCA_DIR, "X_pred.npy")
OUTPUT_PCA_COMP = os.path.join(OUTPUT_PCA_DIR, "components.npy")

##############
# Parameters #
##############

N_COMP = 10
SPARSE_PCA_ALPHA = np.arange(0, 10, 1)

#############
# Functions #
#############

#################
# Actual script #
#################

# Read learning data (french & german subjects)
X = np.load(INPUT_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

# PCA with all data
pca = sklearn.decomposition.PCA(n_components=10)
X_pca = pca.fit(X).transform(X)
np.save(OUTPUT_PCA_PRED, X_pca)
np.save(OUTPUT_PCA_COMP, pca.components_)
