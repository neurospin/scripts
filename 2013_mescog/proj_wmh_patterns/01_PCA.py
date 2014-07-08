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

from sklearn.cross_validation import StratifiedKFold
import sklearn.decomposition

import pandas as pd

import nibabel

import matplotlib.pyplot as plt

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"

INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "mescog", "proj_wmh_patterns")

## We need the original dataset for display
INPUT_DATASET = os.path.join(INPUT_DIR, "X_center.npy")
INPUT_ORIGINAL_DATASET = os.path.join(INPUT_DIR, "X.npy")
INPUT_CSV = os.path.join(INPUT_DIR, "population.csv")
INPUT_MASK = os.path.join(INPUT_DIR, "mask_bin.nii")

OUTPUT_BASE_DIR = INPUT_DIR
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# train will be replaced by the training index
OUTPUT_TRAIN_DIR = os.path.join(OUTPUT_DIR,
                                "{train}")
OUTPUT_TRAIN_ID_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                   "train_id.txt")
OUTPUT_TEST_ID_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                  "test_id.txt")
OUTPUT_PCA_COMP_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                   "PCA.npy")
OUTPUT_TRAIN_PROJ_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                     "X_train_proj.npy")
OUTPUT_TEST_PROJ_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                    "X_test_proj.npy")
OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_TRAIN_DIR,
                                   "{i:03}")
OUTPUT_MIN_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "min.{ID:04}.nii")
OUTPUT_MAX_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "max.{ID:04}.nii")
# TODO: modify output file names
OUTPUT_CSV = os.path.join(OUTPUT_TRAIN_DIR, "pc_learn_{train}.csv")

##############
# Parameters #
##############

N_COMP = 5
N_FOLDS = 2

#############
# Functions #
#############

def compute_project(X_train, X_test):
    # Compute decomposition
    PCA = sklearn.decomposition.PCA()
    PCA.fit(X_train)

    # Project training subjects onto PC
    # scikit-learn projects on min(n_samples, n_features)
    X_train_proj = PCA.transform(X_train)

    # Project learning subjects onto PC
    # scikit-learn projects on min(n_samples, n_features)
    X_test_proj = PCA.transform(X_test)

    return PCA, X_train_proj, X_test_proj

#################
# Actual script #
#################

if __name__ == "__main__":
    # Read clinic status (used to split groups)
    clinic_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
    clinic_subjects_id = clinic_data.index
    print "Found", len(clinic_subjects_id), "clinic records"

    # Stratification of subjects
    y = clinic_data['SITE'].map({'FR': 0, 'GE': 1})
    skf = StratifiedKFold(y=y, n_folds=N_FOLDS)

    # Read mask
    babel_mask = nibabel.load(INPUT_MASK)
    mask = babel_mask.get_data()
    binary_mask = mask != 0

    # Read dataset
    X = np.load(INPUT_DATASET)
    X_orig = np.load(INPUT_ORIGINAL_DATASET)

    #
    PCA = {}
    for train_index, (tr, te) in enumerate(skf):
        print "Learning {i}/{n}".format(i=train_index+1, n=N_FOLDS)
        X_train = X[tr]
        X_test  = X[te]
        train_subjects_id = clinic_data.index[tr]
        test_subjects_id = clinic_data.index[te]
        PCA[train_index], train_proj, test_proj = compute_project(X_train, X_test)

        # Store results
        out_dir = OUTPUT_TRAIN_DIR.format(train=train_index)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Store subjects ID
        with open(OUTPUT_TRAIN_ID_FMT.format(train=train_index), "w") as f:
            for _id in train_subjects_id:
                print >> f, _id
        with open(OUTPUT_TEST_ID_FMT.format(train=train_index), "w") as f:
            for _id in test_subjects_id:
                print >> f, _id

        # Store components
        np.save(OUTPUT_PCA_COMP_FMT.format(train=train_index),
                PCA[train_index].components_)

        # Store projections
        np.save(OUTPUT_TRAIN_PROJ_FMT.format(train=train_index),
                train_proj)
        np.save(OUTPUT_TEST_PROJ_FMT.format(train=train_index),
                test_proj)

        # Save the components into brain-like images
        # We also save the extremum subjects
        for comp_index in range(N_COMP):
            # Create dir
            output_dir = OUTPUT_COMP_DIR_FMT.format(i=comp_index,
                                                    train=train_index)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            comp_data = np.zeros(binary_mask.shape)
            comp_data[:] = np.NaN
            comp_data[binary_mask] = PCA[train_index].components_[comp_index, :]
            comp_im = nibabel.Nifti1Image(comp_data,
                                          babel_mask.get_affine(),
                                          header=babel_mask.get_header())
            name = os.path.join(output_dir, "loading_{i:03}.nii").format(i=comp_index)
            nibabel.save(comp_im, name)
            # Save image of extremum subjects for this component
            extremum_sub = (min_sub, max_sub) = (train_proj[:, comp_index].argmin(),
                                                 train_proj[:, comp_index].argmax())
            names = (OUTPUT_MIN_SUBJECT_FMT.format(train=train_index,
                                                   i=comp_index,
                                                   ID=train_subjects_id[min_sub]),
                     OUTPUT_MAX_SUBJECT_FMT.format(train=train_index,
                                                   i=comp_index,
                                                   ID=train_subjects_id[max_sub]))
            for (index, name) in zip(extremum_sub, names):
                data = np.zeros(binary_mask.shape)
                data[binary_mask] = X_orig[index, :]
                im = nibabel.Nifti1Image(data,
                                         babel_mask.get_affine(),
                                         header=babel_mask.get_header())
                nibabel.save(im, name)

        ####################
        ## Plotting graphs #
        ####################

        # Store and plot percentage of explained variance
        explained_variance = PCA[train_index].explained_variance_ratio_
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

        # Create df file with all subjects (site, ID, PC1, PC2, PC3)
        index = pd.Index(np.hstack((train_subjects_id, test_subjects_id)),
                         name='Subject ID')
        PC0 = pd.Series.from_array(np.hstack((train_proj[:, 0], test_proj[:, 0])),
                                   index=index)
        PC1 = pd.Series.from_array(np.hstack((train_proj[:, 1], test_proj[:, 1])),
                                   index=index)
        PC2 = pd.Series.from_array(np.hstack((train_proj[:, 2], test_proj[:, 2])),
                                   index=index)
        df = pd.DataFrame.from_items((("SITE", y),
                                      ("PC0", PC0),
                                      ("PC1", PC1),
                                      ("PC2", PC2)))
        df.to_csv(OUTPUT_CSV.format(train=train_index))
