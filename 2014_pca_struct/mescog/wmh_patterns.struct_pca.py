# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 10:35:45 2014

@author: md238665

Read the data generated in 2013_mescog/proj_wmh_patterns.

We use the centered data

"""

import os

import numpy as np
import scipy as sp

import pandas as pd

import nibabel

import matplotlib.pyplot as plt

import parsimony
import pca_tv

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

# Output for struct PCA
OUTPUT_STRUCT_PCA_DIR  = os.path.join(OUTPUT_DIR, "StructPCA_{k}_{l}_{g}")
OUTPUT_STRUCT_PCA_PRED = os.path.join(OUTPUT_STRUCT_PCA_DIR, "X_pred.npy")
OUTPUT_STRUCT_PCA_COMP = os.path.join(OUTPUT_STRUCT_PCA_DIR, "components.npy")

##############
# Parameters #
##############

N_COMP = 10
STRUCT_PCA_ALPHA = np.arange(0, 10, 1)

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
linear_mask = np.where(binary_mask)

# PCA with structured constraints
# A matrices
natural_shape = babel_mask.shape
n, p = X.shape
Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_mask(mask)
Al1 = sp.sparse.eye(p, p)

k = 1
l = 1
g = 1
#for alpha in SPARSE_PCA_ALPHA:
if True:
    struct_pca = pca_tv.PCA_SmoothedL1_L2_TV(k, l, g, Atv, Al1, n_components=2)
    X_struct_pca = struct_pca.fit(X).transform(X)
    output_dir = OUTPUT_STRUCT_PCA_DIR.format(alpha=alpha)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(OUTPUT_STRUCT_PCA_PRED.format(k=k,l=l,g=g), X_struct_pca)
    np.save(OUTPUT_STRUCT_PCA_COMP.format(k=k,l=l,g=g), struct_pca.components_)
