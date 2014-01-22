# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:17:26 2014

@author: lh239456
"""
import os
import pickle

import numpy as np

import scipy
import scipy.cluster.hierarchy
import scipy.spatial.distance

import pandas as pd

import nibabel

import matplotlib.pyplot as plt
##################
# Input & output #
##################


INPUT_BASE_DIR = "/neurospin/"
INPUT_CSV_DIR = os.path.join(INPUT_BASE_DIR,
                             "mescog", "proj_predict_cog_decline", "data")
INPUT_CSV = os.path.join(INPUT_CSV_DIR, "dataset_clinic_niglob_20140121.csv")

INPUT_PCA_DIR = os.path.join(INPUT_BASE_DIR,
                             "mescog", "proj_wmh_patterns.back", "PCA")
INPUT_PCA = os.path.join(INPUT_PCA_DIR, "PCA.pkl")

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns.back")
INPUT_TRAIN_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")
INPUT_TRAIN_DATASET = os.path.join(INPUT_DATASET_DIR,
                                   "french.center.npy")
OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns.back", "PCA_plots")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_PLOT_FMT = os.path.join(OUTPUT_DIR, "{feature}_pca{n_comp:02}")

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_TRAIN_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)

# Read subjects ID
with open(INPUT_TRAIN_SUBJECTS) as f:
    TRAIN_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Get info for csv file
data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
csv_subjects_id = [int(subject_id[4:]) for subject_id in data.index]
data.index = csv_subjects_id

TRAIN_SUBJECTS_AGE = data["AGE_AT_INCLUSION"][TRAIN_SUBJECTS_ID]
TRAIN_SUBJECTS_VOLUME = data["BRAINVOL"][TRAIN_SUBJECTS_ID]

# Get PCA model
with open(INPUT_PCA, "rb") as f:
    PCA = pickle.load(f)

# Project subjects onto dimensions
# scikit-learn projects on min(n_samples, n_features)
X_proj = PCA.transform(X)

####################
# Plotting figures #
####################
N_COMP = 10

# For each component of the pca, we plot the distribution of age
# and brain volume
for i in range(N_COMP):
    pca_age_fig = plt.figure()
    plt.scatter(X_proj[:, i], TRAIN_SUBJECTS_AGE)
    NAME = os.path.join(OUTPUT_PLOT_FMT.format(feature="age", n_comp=i))
    plt.savefig(NAME)

    pca_age_fig = plt.figure()
    plt.scatter(X_proj[:, i], TRAIN_SUBJECTS_VOLUME)
    NAME = os.path.join(OUTPUT_PLOT_FMT.format(feature="volume", n_comp=i))
    plt.savefig(NAME)
