# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:39:01 2014

@author: md238665

Do some plots for PCA interpretation.

"""

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

##################
# Input & output #
##################


INPUT_BASE_DIR = "/neurospin/"

INPUT_CLINIC_DIR = os.path.join(INPUT_BASE_DIR,
                                "mescog", "proj_predict_cog_decline", "data")
INPUT_CSV = os.path.join(INPUT_CLINIC_DIR,
                         "dataset_clinic_niglob_20140121.csv")

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns")

# We need the original dataset for display
INPUT_DATASET = {}
INPUT_SUBJECTS = {}
INPUT_ORIG_DATASET = {}
# French datasets
#INPUT_DATASET["FR"] = os.path.join(INPUT_DATASET_DIR,
#                                    "french.center.npy")
#INPUT_ORIG_DATASET["FR"] = os.path.join(INPUT_DATASET_DIR,
#                                        "french.npy")
INPUT_SUBJECTS["FR"] = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")
INPUT_TEST_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                   "germans-subjects.txt")

# Germans datsaets
#INPUT_DATASET["GE"] = os.path.join(INPUT_DATASET_DIR,
#                                   "germans.center.npy")
#INPUT_ORIG_DATASET["GE"] = os.path.join(INPUT_DATASET_DIR,
#                                        "germans.npy")
INPUT_SUBJECTS["GE"] = os.path.join(INPUT_DATASET_DIR,
                                    "germans-subjects.txt")

#INPUT_PCA_COMPONENT = os.path.join(INPUT_PCA_DIR, "PCA.npy")
#INPUT_FRENCH_PROJ = os.path.join(INPUT_PCA_DIR, "french.proj.npy")
#INPUT_GERMANS_PROJ = os.path.join(INPUT_PCA_DIR, "germans.proj.npy")
#
#INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
#                                 "mescog", "proj_wmh_patterns")
#INPUT_FRENCH_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
#                                     "french-subjects.txt")
#INPUT_GERMANS_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
#                                      "germans-subjects.txt")

INPUT_PCA_DIR = os.path.join(INPUT_BASE_DIR,
                             "mescog", "proj_wmh_patterns", "PCA")
# train will be replaced by the training nationality
INPUT_TRAINNAT_DIR = os.path.join(INPUT_PCA_DIR,
                                  "train_{train}")
INPUT_PCA_COMP_FMT = os.path.join(INPUT_TRAINNAT_DIR,
                                  "PCA.npy")
INPUT_TRAIN_PROJ_FMT = os.path.join(INPUT_TRAINNAT_DIR,
                                    "X_train.proj.npy")  # Use train nat?
INPUT_TEST_PROJ_FMT = os.path.join(INPUT_TRAINNAT_DIR,
                                   "X_test.proj.npy")
INPUT_COMP_DIR_FMT = os.path.join(INPUT_TRAINNAT_DIR,
                                  "{i:03}")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog",
                          "proj_wmh_patterns",
                          "PCA",
                          "train_{train}")

OUTPUT_PLOT_FMT = os.path.join(OUTPUT_DIR, "{feature}_pca{n_comp:02}")

##############
# Parameters #
##############

N_COMP = 3

#################
# Actual script #
#################

# Read clinic data
clinic_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
csv_subjects_id = [int(subject_id[4:]) for subject_id in clinic_data.index]
clinic_data.index = csv_subjects_id

SUBJECTS_ID = {}
# Read french subjects ID
with open(INPUT_SUBJECTS["FR"]) as f:
    SUBJECTS_ID["FR"] = np.array([int(l) for l in f.readlines()])
# Read germans subjects ID
with open(INPUT_SUBJECTS["GE"]) as f:
    SUBJECTS_ID["GE"] = np.array([int(l) for l in f.readlines()])

clin_data = {}
# Subsample clinic data for train subjects
clin_data["FR"] = clinic_data.loc[SUBJECTS_ID["FR"]]
clin_data["GE"] = clinic_data.loc[SUBJECTS_ID["GE"]]

NAT = ["FR", "GE"]
LEGEND_NAME = ["French", "Germans"]
PLOT_SYMBOL = ["o", "^"]
NAT_INFO = zip(NAT, LEGEND_NAME, PLOT_SYMBOL)
L = zip(NAT_INFO, reversed(NAT_INFO))
for train_info, test_info in L:
    train_nat, train_legend_name, train_plot_symbol = train_info
    test_nat, test_legend_name, test_plot_symbol = test_info
    print "Figure for learning with %s" % train_nat

    # Load projection of train & test subjects onto PC
    train_proj = np.load(INPUT_TRAIN_PROJ_FMT.format(train=train_nat))
    test_proj = np.load(INPUT_TEST_PROJ_FMT.format(train=train_nat))

    ####################
    # Plotting figures #
    ####################

    # For each component of the PCA, we plot the distribution of age
    # and brain volume
    for i in range(N_COMP):
        # Age distribution
        pca_age_fig = plt.figure()
        plt.scatter(train_proj[:, i], clin_data[train_nat]["AGE_AT_INCLUSION"])
        pca_age_fig.suptitle("Distribution of age along the"
                             " {n_comp}th component".format(n_comp=i+1))
        ax = plt.gca()
        #    ax.spines['left'].set_position('zero')
        #    ax.spines['right'].set_color('none')
        plt.xlabel('PCA_{n_comp}'.format(n_comp=i+1))
        plt.ylabel("Age")
        #    plt.text(x=0, y=92, s='age', horizontalalignment='left',
        #             verticalalignment='bottom')
        NAME = os.path.join(OUTPUT_PLOT_FMT.format(train=train_nat,
                                                   feature="age",
                                                   n_comp=i))
        plt.savefig(NAME)
        # Brain volume distribution
        pca_volume_fig = plt.figure()
        plt.scatter(train_proj[:, i], clin_data[train_nat]["BRAINVOL"])
        pca_volume_fig.suptitle("Distribution of brain volume along the"
                                " {n_comp}th component".format(n_comp=i+1))
        ax = plt.gca()
        #    ax.spines['left'].set_position('zero')
        #    ax.spines['right'].set_color('none')
        plt.xlabel('PCA_{n_comp}'.format(n_comp=i+1))
        plt.ylabel('Brain volume')
        #    plt.text(x=0, y=92, s='brain volume', horizontalalignment='left',
        #             verticalalignment='bottom')
        NAME = OUTPUT_PLOT_FMT.format(train=train_nat,
                                      feature="volume",
                                      n_comp=i)
        plt.savefig(NAME)
    print "age and brain volume graphs saved"

    # Plot of 3rd component of PCA along 2nd component
    pca2_pca3_fig = plt.figure()
    train = plt.scatter(train_proj[:, 1], train_proj[:, 2],
                        marker=train_plot_symbol, s=50)
    test = plt.scatter(test_proj[:, 1], test_proj[:, 2],
                       marker=test_plot_symbol, s=50)
    # Legend
    plt.legend((train, test),
           (train_legend_name, test_legend_name),
           scatterpoints=1,
           loc='upper left')
    plt.xlim([-300, 125])
    plt.ylim([-150, 125])
    pca2_pca3_fig.suptitle("Position of subjects along 2nd and 3rd principal "
                           "components")
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.text(x=42, y=0, s='PCA_2', horizontalalignment='left',
             verticalalignment='center')
    plt.text(x=0, y=42, s='PCA_3', horizontalalignment='left',
             verticalalignment='bottom')
    NAME = os.path.join(OUTPUT_DIR, "pca3_pca2.svg").format(train=train_nat)
    plt.savefig(NAME)
    # Add annotations
    for i, subject_id in enumerate(SUBJECTS_ID[train_nat]):
        plt.annotate(str(subject_id), xy=(train_proj[i, 1], train_proj[i, 2]))
    for i, subject_id in enumerate(SUBJECTS_ID[test_nat]):
        plt.annotate(str(subject_id), xy=(test_proj[i, 1], test_proj[i, 2]))
    NAME = os.path.join(OUTPUT_DIR, "pca3_pca2_annot.svg").format(train=train_nat)
    plt.savefig(NAME)

    # Plot of 3rd component of PCA along 2nd component
    # with color given by clinical variable
    # Warning: subjects for which the color is NaN won't appear.
    VAR = ["BPF", "LLcount", "MDRS_TOTAL", "MRS", "MRS@M36"]
    for var in VAR:
        print "Plotting with", var
        pca2_pca3_fig_colors = plt.figure()
        # Plot french
        colors = clin_data[train_nat][var]
        train = plt.scatter(train_proj[:, 1],
                            train_proj[:, 2],
                            marker=train_plot_symbol, c=colors, s=50)
        train.set_alpha(0.75)
        # Plot germans
        colors = clin_data[test_nat][var]
        test = plt.scatter(test_proj[:, 1],
                           test_proj[:, 2],
                           marker=test_plot_symbol, c=colors, s=50)
        test.set_alpha(0.75)
        # Legend
        plt.legend((train, test),
                   (train_legend_name, test_legend_name),
                   scatterpoints=1,
                   loc='upper left')
        # Colorbar
        cb = plt.colorbar()
        cb.set_label(var)
        # Axes
        plt.xlim([-300, 125])
        plt.ylim([-150, 125])
        pca2_pca3_fig_colors.suptitle("Position of subjects along 2nd and 3rd "
                                      "principal components")
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        plt.text(x=100, y=5, s='PCA_2', horizontalalignment='center',
                 verticalalignment='center')
        plt.text(x=0, y=100, s='PCA_3', horizontalalignment='left',
                 verticalalignment='bottom')
        #for i, subject_id in zip(SUBJECT_INDEX_WITH_LL_COUNT, SUBJECT_ID_WITH_LL_COUNT):
        #    plt.annotate(str(subject_id), xy=(X_proj[i, 1], X_proj[i, 2]))
        #plt.show()

        NAME = os.path.join(OUTPUT_DIR, "pca3_pca2_{var}.svg").format(
          train=train_nat,
          var=var)
        plt.savefig(NAME)
    print "position graphs saved"
