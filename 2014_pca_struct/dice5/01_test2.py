# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:37:33 2014

@author: md238665

Simple test of the algorithm on the dice 5 data.

"""

import os

import numpy as np

import scipy
import scipy.sparse

import matplotlib.pyplot as plt

import sklearn
import sklearn.preprocessing
import sklearn.decomposition

import parsimony
import parsimony.functions
from parsimony.algorithms import *

from parsimony.utils import plot_map2d

import pca_tv

import timeit

# RNG seed to get reproducible results
np.random.seed(seed=13031981)


INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"

INPUT_DIR = os.path.join(INPUT_BASE_DIR, "data")
INPUT_DATASET_FILE_FORMAT = "data_{alpha}.npy"
INPUT_STD_DATASET_FILE_FORMAT = "data_{alpha}.std.npy"
INPUT_BETA_FILE_FORMAT = "beta3d_{alpha}.std.npy"
INPUT_INDEX_FILE_FORMAT = "{subset}_indices.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"

INPUT_SHAPE = (100, 100, 1)
INPUT_ALPHAS = np.array([0.01, 0.1, 1, 10])
INPUT_RESAMPLE = os.path.join(INPUT_DIR, "train_indices.npy")

##############
# Parameters #
##############

N_COMP = 3

########
# Code #
########

# Load data
snr = 0.3
filename = INPUT_STD_DATASET_FILE_FORMAT.format(alpha=snr)
dataset_full_filename = os.path.join(INPUT_DIR, filename)
X = np.load(dataset_full_filename)
filename = INPUT_BETA_FILE_FORMAT.format(alpha=alpha)
beta3d_full_filename = os.path.join(INPUT_DIR, filename)
beta3d = np.load(beta3d_full_filename)

n, p = X.shape

Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_shape(INPUT_SHAPE)
Al1 = scipy.sparse.eye(p, p)

#
# Fit models with l1 constraint
#
alpha = 1
l1 = alpha * 0.4

# PCA
print "Fitting PCA"
pca_sklearn = sklearn.decomposition.PCA(n_components=N_COMP)
pca_sklearn.fit(X)
V_pca_sklearn = pca_sklearn.components_.transpose()
del pca_sklearn

# Sparse PCA
print "Fitting SparsePCA"
sparsepca_sklearn = sklearn.decomposition.SparsePCA(n_components=N_COMP,
                                                    alpha=l1)
t = timeit.timeit(stmt='sparsepca_sklearn.fit(X)', setup="from __main__ import sparsepca_sklearn, X", number=1)
print "Sparse PCA:", t
V_sparsepca_sklearn = sparsepca_sklearn.components_.transpose()
del sparsepca_sklearn

# Struct PCA with few TV
print "Fitting StructPCA"
l2 = 1
ltv = alpha * .001
e1 = pca_tv.PCA_SmoothedL1_L2_TV(l1, l2, ltv, Atv, Al1,
                                 n_components=N_COMP,
                                 criterion="frobenius",
                                 eps=1e-6,
#                                inner_eps=1e-8,
                                 inner_max_iter=int(1e5),
                                 use_eg=False,
                                 output=False)
t = timeit.timeit(stmt='e1.fit(X)', setup="from __main__ import e1, X", number=1)
print "Fitting StructPCA:", t
V1 = e1.V

# Struct PCA with more TV
print "Fitting StructPCA with more TV"
ltv = alpha * .01
e2 = pca_tv.PCA_SmoothedL1_L2_TV(l1, l2, ltv, Atv, Al1,
                                 n_components=N_COMP,
                                 criterion="frobenius",
                                 eps=1e-6,
#                                inner_eps=1e-8,
                                 inner_max_iter=int(1e5),
                                 use_eg=False,
                                 output=False)
t = timeit.timeit(stmt='e2.fit(X)', setup="from __main__ import e2, X", number=1)
print "Fitting StructPCA with more TV", t
V2 = e2.V

plot = plt.subplot(5, N_COMP, 1)
plot_map2d(beta3d.reshape(INPUT_SHAPE), plot, title="beta star")
for k in range(N_COMP):
    # Plot PCA
    plot = plt.subplot(5, N_COMP, N_COMP+1+k)
    title = "PCA comp #{i}".format(i=k)
    plot_map2d(V_pca_sklearn[:,k].reshape(INPUT_SHAPE), plot, title=title)

    # Plot sparsepca_sklearn
    plot = plt.subplot(5, N_COMP, 2*N_COMP+1+k)
    title = "SparsePCA comp #{i}".format(i=k)
    plot_map2d(V_sparsepca_sklearn[:,k].reshape(INPUT_SHAPE), plot, title=title)

    # Plot the components CONESTA
    plot = plt.subplot(5, N_COMP, 3*N_COMP+1+k)
    title = "PCA_L1L2TV CONESTA comp #{i} (TV={tv})".format(i=k,
                                                            tv=e1.ltv)
    plot_map2d(V1[:,k].reshape(INPUT_SHAPE), plot, title=title)

    # Plot the components CONESTA
    plot = plt.subplot(5, N_COMP, 4*N_COMP+1+k)
    title = "PCA_L1L2TV CONESTA comp #{i} (TV={tv})".format(i=k,
                                                            tv=e2.ltv)
    plot_map2d(V2[:,k].reshape(INPUT_SHAPE), plot, title=title)


plt.show()
