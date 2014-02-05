# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2013

@author: md238665

Use PCA to find the most important axes in data.
We create some plots and analysis of the components.

Use centered but not scaled data.

TODO:
 - kernel PCA?

"""

import os

import numpy as np

import sklearn
import sklearn.decomposition

import pandas as pd

import nibabel

import matplotlib.pyplot as plt

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns")

# We need the original dataset for display
INPUT_DATASET = {}
INPUT_SUBJECTS = {}
INPUT_ORIG_DATASET = {}
# French datasets
INPUT_DATASET["FR"] = os.path.join(INPUT_DATASET_DIR,
                                   "french.center.npy")
INPUT_ORIG_DATASET["FR"] = os.path.join(INPUT_DATASET_DIR,
                                        "french.npy")
INPUT_SUBJECTS["FR"] = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")

# Germans datasets
INPUT_DATASET["GE"] = os.path.join(INPUT_DATASET_DIR,
                                   "germans.center.npy")
INPUT_ORIG_DATASET["GE"] = os.path.join(INPUT_DATASET_DIR,
                                        "germans.npy")
INPUT_SUBJECTS["GE"] = os.path.join(INPUT_DATASET_DIR,
                                    "germans-subjects.txt")

INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# train will be replaced by the training nationality
OUTPUT_TRAINNAT_DIR = os.path.join(OUTPUT_DIR,
                                   "train_{train}")
OUTPUT_PCA_COMP_FMT = os.path.join(OUTPUT_TRAINNAT_DIR,
                                   "PCA.npy")
OUTPUT_TRAIN_PROJ_FMT = os.path.join(OUTPUT_TRAINNAT_DIR,
                                     "X_train.proj.npy") # Use train nat?
OUTPUT_TEST_PROJ_FMT = os.path.join(OUTPUT_TRAINNAT_DIR,
                                    "X_test.proj.npy")
OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_TRAINNAT_DIR,
                                   "{i:03}")
OUTPUT_MIN_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "min.{ID:04}.nii")
OUTPUT_MAX_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "max.{ID:04}.nii")
# TODO: modify output file names
OUTPUT_CSV = os.path.join(OUTPUT_TRAINNAT_DIR, "pc_learn_{train}.csv")

##############
# Parameters #
##############

N_COMP = 10

#############
# Functions #
#############


def compute_project(X_train, X_test):
    # Compute decomposition
    PCA = sklearn.decomposition.PCA()
    PCA.fit(X_train)

    # Project french subjects onto PC & save it
    # scikit-learn projects on min(n_samples, n_features)
    X_train_proj = PCA.transform(X_train)

    # Project german subjects onto PC
    # scikit-learn projects on min(n_samples, n_features)
    X_test_proj = PCA.transform(X_test)

    return PCA, X_train_proj, X_test_proj

#################
# Actual script #
#################

X = {}
SUBJECTS_ID = {}

# Read learning data (french & german subjects)
X["FR"] = np.load(INPUT_DATASET["FR"])
print "Data loaded: {s[0]}x{s[1]}".format(s=X["FR"].shape)

with open(INPUT_SUBJECTS["FR"]) as f:
    SUBJECTS_ID["FR"] = np.array([int(l) for l in f.readlines()])

# Read test data (german subjects)
X["GE"] = np.load(INPUT_DATASET["GE"])
print "Data loaded: {s[0]}x{s[1]}".format(s=X["GE"].shape)

with open(INPUT_SUBJECTS["GE"]) as f:
    SUBJECTS_ID["GE"] = np.array([int(l) for l in f.readlines()])

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

#
PCA = {}
for train_nat, test_nat in zip(["FR", "GE"], ["GE", "FR"]):
    print "Learning with %s (testing with %s)" % (train_nat, test_nat)
    X_train = X[train_nat]
    X_test  = X[test_nat]
    TRAIN_SUBJECTS_ID = SUBJECTS_ID[train_nat]
    TEST_SUBJECTS_ID  = SUBJECTS_ID[test_nat]
    PCA[train_nat], train_proj, test_proj = compute_project(X_train, X_test)

    # Store results
    out_dir = OUTPUT_TRAINNAT_DIR.format(train=train_nat)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # Store components
    np.save(OUTPUT_PCA_COMP_FMT.format(train=train_nat), PCA[train_nat].components_)
    # Store projections
    np.save(OUTPUT_TRAIN_PROJ_FMT.format(train=train_nat),
            train_proj)
    np.save(OUTPUT_TEST_PROJ_FMT.format(train=train_nat),
            test_proj)

    # Save the components into brain-like images
    # We also save the extremum subjects
    # Read the whole train dataset
    X_orig = np.load(INPUT_ORIG_DATASET[train_nat])
    for i in range(N_COMP):
        # Create dir
        output_dir = OUTPUT_COMP_DIR_FMT.format(i=i, train=train_nat)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        comp_data = np.zeros(binary_mask.shape)
        comp_data[:] = np.NaN
        comp_data[binary_mask] = PCA[train_nat].components_[i, :]
        comp_im = nibabel.Nifti1Image(comp_data,
                                      babel_mask.get_affine(),
                                      header=babel_mask.get_header())
        name = os.path.join(output_dir, "loading_{i:03}.nii").format(i=i)
        nibabel.save(comp_im, name)
        # Save image of extremum subjects for this component
        extremum_sub = (min_sub, max_sub) = (train_proj[:, i].argmin(), train_proj[:, i].argmax())
        names = (OUTPUT_MIN_SUBJECT_FMT.format(train=train_nat,
                                               i=i,
                                               ID=TRAIN_SUBJECTS_ID[min_sub]),
                 OUTPUT_MAX_SUBJECT_FMT.format(train=train_nat,
                                               i=i,
                                               ID=TRAIN_SUBJECTS_ID[max_sub]))
        for (index, name) in zip(extremum_sub, names):
            data = np.zeros(binary_mask.shape)
            data[binary_mask] = X_orig[index, :]
            im = nibabel.Nifti1Image(data,
                                     babel_mask.get_affine(),
                                     header=babel_mask.get_header())
            nibabel.save(im, name)
    del X_orig

    ####################
    ## Plotting graphs #
    ####################

    # Store and plot percentage of explained variance
    explained_variance = PCA[train_nat].explained_variance_ratio_
    filename = os.path.join(out_dir,
                            "explained_variance.txt")
    np.savetxt(filename, explained_variance)
    explained_variance_cumsum = explained_variance.cumsum()
    filename = os.path.join(out_dir,
                            "explained_variance.cumsum.txt")
    np.savetxt(filename, explained_variance_cumsum)
    x_max = explained_variance.shape[0] + 1

    explained_variance_fig = plt.figure()
    plt.plot(range(1, x_max), explained_variance)
    explained_variance_fig.suptitle('Ratio of explained variance')
    plt.xlabel('Rank')
    plt.ylabel('Explained variance ratio')
    filename = os.path.join(out_dir,
                            "explained_variance.png")
    plt.savefig(filename)
    # Zoom in [1, N_COMP+1]
    axes = explained_variance_fig.axes
    axes[0].set_xlim([1, N_COMP+1])
    axes[0].set_ylim([explained_variance[N_COMP], explained_variance[0]+0.05])
    plt.xticks(np.arange(1, N_COMP+1, 1.0))
    filename = os.path.join(out_dir,
                            "explained_variance.zoom.png")
    plt.savefig(filename)

    explained_variance_cumsum_fig = plt.figure()
    plt.plot(range(1, x_max), explained_variance_cumsum)
    explained_variance_cumsum_fig.suptitle('Ratio of explained variance (cumsum)')
    plt.xlabel('Rank')
    plt.ylabel('Explained variance ratio')
    filename = os.path.join(out_dir,
                            "explained_variance_cumsum.png")
    plt.savefig(filename)
    # Zoom in [1, N_COMP+1]
    axes = explained_variance_cumsum_fig.axes
    axes[0].set_xlim([1, N_COMP+1])
    axes[0].set_ylim([explained_variance_cumsum[0], explained_variance_cumsum[N_COMP]+0.05])
    plt.xticks(np.arange(1, N_COMP+1, 1.0))
    filename = os.path.join(out_dir,
                            "explained_variance_cumsum.zoom.png")
    plt.savefig(filename)

    ###############################################
    # Create csv file for interoperability with R #
    ###############################################

    # Create csv file with all subjects (site, ID, PC1, PC2, PC3)
    ALL_SUBJECTS_ID = np.append(TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID)
    #site_mask = ALL_SUBJECTS_ID<2000
    SITE = range(len(ALL_SUBJECTS_ID))
    for i, subject_id in enumerate(ALL_SUBJECTS_ID):
        if subject_id<2000: SITE[i] = 'FR'
        else: SITE[i] = 'GE'
    PC1_TRAIN = train_proj[:, 0]
    PC2_TRAIN = train_proj[:, 1]
    PC3_TRAIN = train_proj[:, 2]
    PC1_TEST = test_proj[:, 0]
    PC2_TEST = test_proj[:, 1]
    PC3_TEST = test_proj[:, 2]
    PC1 = np.append(PC1_TRAIN, PC1_TEST)
    PC2 = np.append(PC2_TRAIN, PC2_TEST)
    PC3 = np.append(PC3_TRAIN, PC3_TEST)
    dataframe = pd.DataFrame({"SITE":SITE, "ID":ALL_SUBJECTS_ID, "PC1":PC1,
                              "PC2":PC2, "PC3":PC3})
    dataframe.to_csv(OUTPUT_CSV.format(train=train_nat))