# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:22:38 2014

@author: lh239456
"""

import os
import pickle

import numpy as np

import sklearn
import sklearn.cluster

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "results", "wmh_patterns")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
#INPUT_SUBJECTS_DIR = os.path.join(INPUT_BASE_DIR,
#                                  "mescog", "datasets")
#INPUT_SUBJECTS = os.path.join(INPUT_SUBJECTS_DIR,
#                              "CAD-WMH-MNI-subjects.txt")
#INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results", "wmh_patterns",
                          "clustering", "Ward")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_DIR, "{i:03}")
OUTPUT_CACHE = os.path.join(OUTPUT_DIR, "tree")

##############
# Parameters #
##############

n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
n, p = s = X.shape
print "Data loaded {s}".format(s=s)

MODELS = []
for nb in n_clusters:
    output_dir = OUTPUT_COMP_DIR_FMT.format(i=nb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print "Trying {nb} cluster(s)".format(nb=nb)
    model = sklearn.cluster.Ward(n_clusters=nb,
                                 memory=OUTPUT_CACHE)
    model.fit(X)
    MODELS.append(model)
    # Save model
    filename = os.path.join(output_dir, "model.pkl")
    with open(filename, "wb") as f:
        pickle.dump(model, f)

print "all files saved"
