# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:57:39 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

Generate test data for PCA with various SNR (controlled by INPUT_ALPHAS).

We split the data into a train and test sets just for future usage.

"""
import os
import numpy as np
from parsimony import datasets
import dice5_pca

INPUT_SHAPE = (100, 100, 1)
INPUT_N_SAMPLES = 100
# Variance of various objects
INPUT_STDEV = np.asarray([2, 1, 0.5])
# SNR
INPUT_ALPHAS = np.asarray([0.01, 0.1, 1, 10])

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/tv_dice5"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "data")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_DATA_FORMAT = "{subset}_{alpha}.npy"
OUTPUT_COMPONENT_MASK_FORMAT = "mask_{i}.npy"
OUTPUT_FULL_MASK = "mask.npy"

for alpha in INPUT_ALPHAS:
    std = alpha * INPUT_STDEV
    objects = dice5_pca.dice_five_with_union_of_pairs(INPUT_SHAPE, std)
    # Generate data
    X3d, _, _ = datasets.regression.dice5.load(n_samples=2*INPUT_N_SAMPLES,
                                               shape=INPUT_SHAPE,
                                               objects=objects,
                                               random_seed=1)
    for i, subset in enumerate(["train", "test"]):
        filename = OUTPUT_DATA_FORMAT.format(alpha=alpha, subset=subset)
        full_filename = os.path.join(OUTPUT_DIR, filename)
        indices = np.arange(i*INPUT_N_SAMPLES, (i+1)*INPUT_N_SAMPLES)
        np.save(full_filename, X3d[indices])
# Generate mask per latent & total mask
import matplotlib.pyplot as plt
full_mask = np.zeros(INPUT_SHAPE, dtype=bool)
for i, o in enumerate(objects):
    mask = o.get_mask()
    full_mask[mask] = True
    cax = plt.matshow(mask.squeeze())
    plt.colorbar(cax)
    plt.title("Mask of latent {i}".format(i=i))
    plt.show()
    filename = OUTPUT_COMPONENT_MASK_FORMAT.format(i=i)
    full_filename = os.path.join(OUTPUT_DIR, filename)
    np.save(full_filename, mask)
filename = os.path.join(OUTPUT_DIR, OUTPUT_FULL_MASK)
np.save(filename, full_mask)
cax = plt.matshow(full_mask.squeeze())
plt.colorbar(cax)
plt.title("Mask")
plt.show()
   # run /home/ed203246/git/scripts/2014_pca_struct/tv_dice5/dice5_pca.py
