# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 11:58:49 2015

@author: fh235918

This script explores a large range of SNR vales. The goal is to find when PCA
starts to have difficulties.
"""
import numpy as np

import scipy

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

from parsimony import datasets
from parsimony.utils import plot_map2d

import timeit

##############
# Parameters #
##############

# RNG seed to get reproducible results
np.random.seed(seed=13031981)

N_COMP = 4
N_SAMPLES = 500
SHAPE = (50, 50, 1)
N_FOLDS = 10

SNR = np.linspace(0.01, 0.5, 50)

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

frobenius_dst_fold = np.zeros((len(SNR), len(range(N_FOLDS))))
evr_fold = np.zeros((len(SNR), len(range(N_FOLDS))))
for i, snr in enumerate(SNR):
    # Create dataset for this SNR value
    print snr
    model = dict(
        # All points has an independant latent
        l1=0., l2=0., l3=1. * snr, l4=0., l5=0.,
        # No shared variance
        l12=1.1 * snr, l45=1. * snr, l12345=0.,
        # Five dots contribute equally
        b1=1., b2=1., b3=1., b4=1., b5=1.)

    X3d, y, beta3d = datasets.regression.dice5.load(
        n_samples=N_SAMPLES, shape=SHAPE,
        sigma_spatial_smoothing=1,
        model=model)

    X = X3d.reshape((N_SAMPLES, np.prod(SHAPE)))

    # Preprocessing
    X = scale(X, axis=0, with_mean=True, with_std=False)

    # Compute score by CV
    pca = PCA(n_components=N_COMP)
    cv = KFold(N_SAMPLES, n_folds=N_FOLDS)
    for j, (train_mask, test_mask) in enumerate(cv):
        X_train = X[train_mask]
        X_test = X[test_mask]
        pca.fit(X_train)
        X_pred = predict_with_pca(pca, X_test)
        frobenius_dst_fold[i, j] = frobenius_dst_score(X_test, X_pred)
        evr_fold[i, j] = pca.explained_variance_ratio_.sum()

frobenius_dst_cv = frobenius_dst_fold.mean(axis=1)
evr_cv = evr_fold.mean(axis=1)
plt.plot(SNR, frobenius_dst_cv)
#plt.plot(SNR, evr_cv)
plt.show()
