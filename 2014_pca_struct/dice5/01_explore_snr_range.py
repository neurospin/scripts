# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:58:49 2015

@author: fh235918

This script explores a large range of SNR vales. The goal is to find when PCA
starts to have difficulties.
"""
import os
import pickle

import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

from parsimony import datasets
from parsimony.utils import plot_map2d

from brainomics import array_utils

import dice5_data
import metrics
import dice5_metrics

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/data"
INPUT_MASK_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/data/masks"
INPUT_DATA_DIR_FORMAT = os.path.join(INPUT_BASE_DIR,
                                     "data_{s[0]}_{s[1]}_{snr}")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{i}.npy"
INUT_MASK_FILE = "mask.npy"
OUTPUT_L1MASK_FILE = "l1_max.txt"

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/calibrate"
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR, "{snr}")
OUTPUT_PCA_FILE = "model.pkl"
OUTPUT_PRED_FILE = "X_pred.npy"

##############
# Parameters #
##############

N_COMP = 3

L2_THRESHOLD = 0.99

TRAIN_RANGE = range(dice5_data.N_SAMPLES/2)
TEST_RANGE = range(dice5_data.N_SAMPLES/2, dice5_data.N_SAMPLES)

# Chosen values (obtained after inspection of results)
# Close to 0.05, 0.1 and 0.2
SNR_TO_DISPLAY = [dice5_data.ALL_SNRS[4],
                  dice5_data.ALL_SNRS[9],
                  dice5_data.ALL_SNRS[19]]

#############
# Functions #
#############


def predict_with_pca(pca, X):
    return np.dot(pca.transform(X), pca.components_)


def frobenius_dst_score(X_test, X_pred, **kwargs):
    n = X_pred.shape[0]
    return np.linalg.norm(X_test - X_pred, ord='fro') / (2 * n)

##########
# Script #
##########

if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR)

# Load masks
sub_objects = []
for i in range(3):
    filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(i=i)
    full_filename = os.path.join(INPUT_MASK_DIR, filename)
    sub_objects.append(np.load(full_filename))

frobenius_dst = np.zeros((len(dice5_data.ALL_SNRS),))
evr = np.zeros((len(dice5_data.ALL_SNRS),))
dice = np.zeros((len(dice5_data.ALL_SNRS), len(range(N_COMP))))
correlation = np.zeros((len(dice5_data.ALL_SNRS), len(range(N_COMP))))
for i, snr in enumerate(dice5_data.ALL_SNRS):
    # Load data
    input_dir = INPUT_DATA_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                             snr=snr)
    std_data_path = os.path.join(input_dir,
                                 INPUT_STD_DATASET_FILE)
    X = np.load(std_data_path)

    # Create output dir
    output_dir = OUTPUT_DIR_FORMAT.format(snr=snr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create dataset for this SNR value
    print snr
    model = dice5_data.create_model(snr)

    # Fit model & compute score
    pca = PCA(n_components=N_COMP)
    X_train = X[TRAIN_RANGE, :]
    X_test = X[TEST_RANGE, :]
    pca.fit(X_train)
    full_filename = os.path.join(output_dir,
                                 OUTPUT_PCA_FILE)
    with open(full_filename, "w") as f:
        pickle.dump(pca, f)
    X_pred = predict_with_pca(pca, X_test)
    full_filename = os.path.join(output_dir,
                                 OUTPUT_PRED_FILE)
    np.save(full_filename, X_pred)
    frobenius_dst[i] = frobenius_dst_score(X_test, X_pred)
    evr[i] = pca.explained_variance_ratio_.sum()
    for j, obj in zip(range(N_COMP), sub_objects):
        _, t = array_utils.arr_threshold_from_norm2_ratio(
            pca.components_[j, :],
            L2_THRESHOLD)
        thresh_comp = pca.components_[j, :] > t
        dice[i, j] = dice5_metrics.dice(thresh_comp.reshape(dice5_data.SHAPE),
                                        obj)
        correlation[i, j] = \
            np.abs(np.corrcoef(pca.components_[j, :],
                               obj.ravel())[1, 0])

plt.figure()
plt.plot(dice5_data.ALL_SNRS, frobenius_dst)
plt.legend(['Frobenius distance'])

for j in range(N_COMP):
    plt.figure()
    legend = 'Correlation between loading #{j} and object #{j}'.format(j=j+1)
    plt.plot(dice5_data.ALL_SNRS, correlation[:, j])
    plt.legend([legend])

# Reload some weight maps
for snr in SNR_TO_DISPLAY:
    output_dir = OUTPUT_DIR_FORMAT.format(snr=snr)
    full_filename = os.path.join(output_dir,
                                 OUTPUT_PCA_FILE)
    with open(full_filename) as f:
        pca = pickle.load(f)

    for j in range(N_COMP):
        legend = 'Loading #{j} for model {snr}'.format(j=j+1,
                                                       snr=snr)
        plot_map2d(pca.components_[j, ].reshape(dice5_data.SHAPE),
                   title=legend)

plt.show()
