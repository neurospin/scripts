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

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/calibrate"

OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR, "{snr}")
OUTPUT_DATASET_FILE = "data.npy"
OUTPUT_STD_DATASET_FILE = "data.std.npy"
OUTPUT_PCA_FILE = "model.pkl"
OUTPUT_PRED_FILE = "X_pred.npy"

##############
# Parameters #
##############

N_COMP = 3
N_TRAIN = 300

L2_THRESHOLD = 0.99

SNR = np.linspace(0.01, 0.5, 50)
# Chosen values (obtained after inspection of results)
# Close to 0.05, 0.1 and 0.2
SNR_TO_DISPLAY = [SNR[4], SNR[9], SNR[19]]

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

train_range = range(N_TRAIN)
test_range = range(N_TRAIN, dice5_data.N_SAMPLES)

frobenius_dst = np.zeros((len(SNR),))
evr = np.zeros((len(SNR),))
dice = np.zeros((len(SNR), len(range(N_COMP))))
correlation = np.zeros((len(SNR), len(range(N_COMP))))
for i, snr in enumerate(SNR):
    output_dir = OUTPUT_DIR_FORMAT.format(snr=snr)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create dataset for this SNR value
    print snr
    model = dice5_data.create_model(snr)

    X3d, y, beta3d = datasets.regression.dice5.load(
        n_samples=dice5_data.N_SAMPLES, shape=dice5_data.SHAPE,
        model=model,
        random_seed=dice5_data.SEED)
    objects = datasets.regression.dice5.dice_five_with_union_of_pairs(
        dice5_data.SHAPE)
    _, _, d3, _, _, union12, union45, _ = objects
    sub_objects = [union12, union45, d3]

    X = X3d.reshape((dice5_data.N_SAMPLES, np.prod(dice5_data.SHAPE)))
    full_filename = os.path.join(output_dir,
                                 OUTPUT_DATASET_FILE)
    np.save(full_filename, X)

    # Preprocessing
    X = scale(X, axis=0, with_mean=True, with_std=False)
    full_filename = os.path.join(output_dir,
                                 OUTPUT_STD_DATASET_FILE)
    np.save(full_filename, X)

    # Fit model & compute score
    pca = PCA(n_components=N_COMP)
    X_train = X[train_range, :]
    X_test = X[test_range, :]
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
                                        obj.get_mask())
        correlation[i, j] = \
            np.abs(np.corrcoef(pca.components_[j, :],
                               obj.get_mask().ravel())[1, 0])

plt.figure()
plt.plot(SNR, frobenius_dst)
plt.legend(['Frobenius distance'])

for j in range(N_COMP):
    plt.figure()
    legend = 'Correlation between loading #{j} and object #{j}'.format(j=j+1)
    plt.plot(SNR, correlation[:, j])
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
