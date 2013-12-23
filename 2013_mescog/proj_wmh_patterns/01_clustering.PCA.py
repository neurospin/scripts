# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2013

@author: md238665

Use PCA to find the most important axes in data.

TODO:
 - kernel PCA?
 - interpretation?

"""

import os
import pickle

import numpy as np

import sklearn
import sklearn.decomposition

##################
# Input & output #
##################

INPUT_BASE_DIR = "/volatile/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "results")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
INPUT_SUBJECTS_DIR = os.path.join(INPUT_BASE_DIR,
                                  "mescog", "datasets")
INPUT_SUBJECTS = os.path.join(INPUT_SUBJECTS_DIR,
                              "CAD-WMH-MNI-subjects.txt")

OUTPUT_BASE_DIR = "/volatile/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PCA = os.path.join(OUTPUT_DIR, "PCA.pkl")

##############
# Parameters #
##############

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)

PCA = sklearn.decomposition.PCA()
PCA.fit(X)
with open(OUTPUT_PCA, "wb") as f:
    pickle.dump(PCA, f)
