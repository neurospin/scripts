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
INPUT_CSV = os.path.join(INPUT_CLINIC_DIR, "dataset_clinic_niglob_20140121.csv")

INPUT_PCA_DIR = os.path.join(INPUT_BASE_DIR,
                             "mescog", "proj_wmh_patterns", "PCA")
INPUT_PCA_COMPONENT = os.path.join(INPUT_PCA_DIR, "PCA.npy")
INPUT_FRENCH_PROJ = os.path.join(INPUT_PCA_DIR, "french.proj.npy")
INPUT_GERMANS_PROJ = os.path.join(INPUT_PCA_DIR, "germans.proj.npy")

INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns")
INPUT_FRENCH_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                     "french-subjects.txt")
INPUT_GERMANS_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                      "germans-subjects.txt")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

# Read french subjects ID
with open(INPUT_FRENCH_SUBJECTS) as f:
    FRENCH_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])
# Read germans subjects ID
with open(INPUT_GERMANS_SUBJECTS) as f:
    GERMANS_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Subsample clinic data for train subjects
french_clinic_data = clinic_data.loc[FRENCH_SUBJECTS_ID]
germans_clinic_data = clinic_data.loc[GERMANS_SUBJECTS_ID]

# Load projection of french subjects onto PC
french_proj = np.load(INPUT_FRENCH_PROJ)
germans_proj = np.load(INPUT_GERMANS_PROJ)

####################
# Plotting figures #
####################

# For each component of the PCA, we plot the distribution of age
# and brain volume
for i in range(N_COMP):
    # Age distribution
    pca_age_fig = plt.figure()
    plt.scatter(french_proj[:, i], french_clinic_data["AGE_AT_INCLUSION"])
    pca_age_fig.suptitle("Distribution of age along the"
                         " {n_comp}th component".format(n_comp=i+1))
    ax = plt.gca()
    #    ax.spines['left'].set_position('zero')
    #    ax.spines['right'].set_color('none')
    plt.xlabel('PCA_{n_comp}'.format(n_comp=i+1))
    plt.ylabel("Age")
    #    plt.text(x=0, y=92, s='age', horizontalalignment='left',
    #             verticalalignment='bottom')
    NAME = os.path.join(OUTPUT_PLOT_FMT.format(feature="age", n_comp=i))
    plt.savefig(NAME)
    # Brain volume distribution
    pca_volume_fig = plt.figure()
    plt.scatter(french_proj[:, i], french_clinic_data["BRAINVOL"])
    pca_volume_fig.suptitle("Distribution of brain volume along the"
                            " {n_comp}th component".format(n_comp=i+1))
    ax = plt.gca()
    #    ax.spines['left'].set_position('zero')
    #    ax.spines['right'].set_color('none')
    plt.xlabel('PCA_{n_comp}'.format(n_comp=i+1))
    plt.ylabel('Brain volume')
    #    plt.text(x=0, y=92, s='brain volume', horizontalalignment='left',
    #             verticalalignment='bottom')
    NAME = OUTPUT_PLOT_FMT.format(feature="volume", n_comp=i)
    plt.savefig(NAME)
print "age and brain volume graphs saved"

# Plot of 3rd component of PCA along 2nd component
pca2_pca3_fig = plt.figure()
fr = plt.scatter(french_proj[:, 1], french_proj[:, 2])
ge = plt.scatter(germans_proj[:, 1], germans_proj[:, 2], marker='^', s=50)
# Legend
plt.legend((fr, ge),
       ('French', 'Germans'),
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
NAME = os.path.join(OUTPUT_DIR, "pca3_pca2.svg")
plt.savefig(NAME)
# Add annotations
for i, subject_id in enumerate(FRENCH_SUBJECTS_ID):
    plt.annotate(str(subject_id), xy=(french_proj[i, 1], french_proj[i, 2]))
for i, subject_id in enumerate(GERMANS_SUBJECTS_ID):
    plt.annotate(str(subject_id), xy=(germans_proj[i, 1], germans_proj[i, 2]))
NAME = os.path.join(OUTPUT_DIR, "pca3_pca2_annot.svg")
plt.savefig(NAME)

# Plot of 3rd component of PCA along 2nd component
# with color given by clinical variable
# Warning: subjects for which the color is NaN won't appear.
VAR = ["BPF", "LLcount", "MDRS_TOTAL", "MRS", "MRS@M36"]
for var in VAR:
    print "Plotting with", var
    pca2_pca3_fig_colors = plt.figure()
    # Plot french
    colors = french_clinic_data[var]
    fr = plt.scatter(french_proj[:, 1], french_proj[:, 2], c=colors, s=50)
    fr.set_alpha(0.75)
    # Plot germans
    colors = germans_clinic_data[var]
    ge = plt.scatter(germans_proj[:, 1], germans_proj[:, 2], marker='^', c=colors, s=50)
    ge.set_alpha(0.75)
    # Legend
    plt.legend((fr, ge),
           ('French', 'Germans'),
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
    
    NAME = os.path.join(OUTPUT_DIR, "pca3_pca2_{var}.svg".format(var=var))
    plt.savefig(NAME)
print "position graphs saved"
