# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

Generate test data for PCA with various SNR (controlled by INPUT_ALPHAS).

We split the data into a train and test sets just for future usage.
Train and test indices are fixed here.

Finally center data.

"""

import os
import numpy as np

from sklearn.preprocessing import StandardScaler

from parsimony import datasets

import dice5_pca

################
# Input/Output #
################

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "data")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DATASET_FILE_FORMAT = "data_{alpha}.npy"
OUTPUT_STD_DATASET_FILE_FORMAT = "data_{alpha}.std.npy"
OUTPUT_BETA_FILE_FORMAT = "beta3d_{alpha}.std.npy"
OUTPUT_INDEX_FILE_FORMAT = "{subset}_indices.npy"
OUTPUT_OBJECT_MASK_FILE_FORMAT = "mask_{i}.npy"
OUTPUT_MASK_FILE_FORMAT = "mask.npy"

##############
# Parameters #
##############

SHAPE = (100, 100, 1)
N_SAMPLES = 100
# Variance of various objects
STDEV = np.asarray([2, 1, 0.5])
# SNR
ALPHAS = np.asarray([0.01, 0.1, 1, 10])

########
# Code #
########

# Generate data for various alpha parameter
for alpha in ALPHAS:
    std = alpha * STDEV
    objects = dice5_pca.dice_five_with_union_of_pairs(SHAPE, std)
    X3d, y, beta3d = datasets.regression.dice5.load(n_samples=2*N_SAMPLES,
                                                    shape=SHAPE,
                                                    objects=objects, random_seed=1)
    # Save data and scaled data
    X = X3d.reshape(2*N_SAMPLES, np.prod(SHAPE))
    filename = OUTPUT_DATASET_FILE_FORMAT.format(alpha=alpha)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, X)
    scaler = StandardScaler(with_mean=True, with_std=False)
    X_std = scaler.fit_transform(X)
    filename = OUTPUT_STD_DATASET_FILE_FORMAT.format(alpha=alpha)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, X_std)
    # Save beta
    filename = OUTPUT_BETA_FILE_FORMAT.format(alpha=alpha)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, beta3d)

# Split in train/test
for i, subset_name in enumerate(["train", "test"]):
    indices = np.arange(i*N_SAMPLES, (i+1)*N_SAMPLES)
    filename = OUTPUT_INDEX_FILE_FORMAT.format(subset=subset_name)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, indices)

# Generate mask with the lastest objects since they have the same geometry
full_mask = np.zeros(SHAPE, dtype=bool)
for i, o in enumerate(objects):
    mask = o.get_mask()
    full_mask += mask
    filename = OUTPUT_OBJECT_MASK_FILE_FORMAT.format(i=i)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, mask)
full_filename = os.path.join(OUTPUT_DIR, OUTPUT_MASK_FILE_FORMAT)
np.save(full_filename, full_mask)
