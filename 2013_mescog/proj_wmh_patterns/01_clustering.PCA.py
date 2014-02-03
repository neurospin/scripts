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
INPUT_TRAIN_DATASET = os.path.join(INPUT_DATASET_DIR,
                                   "french.center.npy")
# We need the original dataset for display
INPUT_ORIG_DATASET = os.path.join(INPUT_DATASET_DIR,
                                  "french.npy")
INPUT_TEST_DATASET = os.path.join(INPUT_DATASET_DIR,
                                   "germans.center.npy")
INPUT_TRAIN_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")
INPUT_TEST_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                   "germans-subjects.txt")
INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PCA_COMP = os.path.join(OUTPUT_DIR, "PCA.npy")
OUTPUT_TRAIN_PROJ = os.path.join(OUTPUT_DIR, "french.proj.npy")
OUTPUT_TEST_PROJ = os.path.join(OUTPUT_DIR, "germans.proj.npy")
OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_DIR, "{i:03}")
OUTPUT_MIN_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "min.{ID:04}.nii")
OUTPUT_MAX_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "max.{ID:04}.nii")
OUTPUT_PLOT_FMT = os.path.join(OUTPUT_DIR, "{feature}_pca{n_comp:02}")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pc_learn_fr.csv")

##############
# Parameters #
##############

N_COMP = 10

#################
# Actual script #
#################

# Read learning data (french subjects)
X_train = np.load(INPUT_TRAIN_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X_train.shape)

# Read test data (german subjects)
X_test = np.load(INPUT_TEST_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X_test.shape)

# Read whole dataset
X_orig = np.load(INPUT_ORIG_DATASET)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

# Read french subjects ID
with open(INPUT_TRAIN_SUBJECTS) as f:
    TRAIN_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])
# Read german subjects ID
with open(INPUT_TEST_SUBJECTS) as f:
    TEST_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Compute decomposition
PCA = sklearn.decomposition.PCA()
PCA.fit(X_train)

# Store components
np.save(OUTPUT_PCA_COMP, PCA.components_)

# Project french subjects onto PC & save it
# scikit-learn projects on min(n_samples, n_features)
X_proj_fr = PCA.transform(X_train)
np.save(OUTPUT_TRAIN_PROJ, X_proj_fr)

# Project german subjects onto PC
# scikit-learn projects on min(n_samples, n_features)
X_proj_ge = PCA.transform(X_test)
np.save(OUTPUT_TEST_PROJ, X_proj_ge)

# Save the components into brain-like images
# We also save the extremum subjects
for i in range(N_COMP):
    # Create dir
    output_dir = OUTPUT_COMP_DIR_FMT.format(i=i)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    comp_data = np.zeros(binary_mask.shape)
    comp_data[:] = np.NaN
    comp_data[binary_mask] = PCA.components_[i, :]
    comp_im = nibabel.Nifti1Image(comp_data, babel_mask.get_affine())
    name = os.path.join(output_dir, "component.nii")
    nibabel.save(comp_im, name)
    # Save image of extremum subjects for this component
    extremum_sub = (min_sub, max_sub) = (X_proj_fr[:, i].argmin(), X_proj_fr[:, i].argmax())
    names = (OUTPUT_MIN_SUBJECT_FMT.format(i=i, ID=TRAIN_SUBJECTS_ID[min_sub]),
             OUTPUT_MAX_SUBJECT_FMT.format(i=i, ID=TRAIN_SUBJECTS_ID[max_sub]))
    for (index, name) in zip(extremum_sub, names):
        data = np.zeros(binary_mask.shape)
        data[binary_mask] = X_orig[index, :]
        im = nibabel.Nifti1Image(data, babel_mask.get_affine())
        nibabel.save(im, name)

###################
# Plotting graphs #
###################

# Store and plot percentage of explained variance
explained_variance = PCA.explained_variance_ratio_
filename = os.path.join(OUTPUT_DIR,
                        "explained_variance.txt")
np.savetxt(filename, explained_variance)
explained_variance_cumsum = explained_variance.cumsum()
filename = os.path.join(OUTPUT_DIR,
                        "explained_variance.cumsum.txt")
np.savetxt(filename, explained_variance_cumsum)
x_max = explained_variance.shape[0] + 1

explained_variance_fig = plt.figure()
plt.plot(range(1, x_max), explained_variance)
explained_variance_fig.suptitle('Ratio of explained variance')
plt.xlabel('Rank')
plt.ylabel('Explained variance ratio')
filename = os.path.join(OUTPUT_DIR,
                        "explained_variance.png")
plt.savefig(filename)
# Zoom in [1, N_COMP+1]
axes = explained_variance_fig.axes
axes[0].set_xlim([1, N_COMP+1])
axes[0].set_ylim([explained_variance[N_COMP], explained_variance[0]+0.05])
plt.xticks(np.arange(1, N_COMP+1, 1.0))
filename = os.path.join(OUTPUT_DIR,
                        "explained_variance.zoom.png")
plt.savefig(filename)

explained_variance_cumsum_fig = plt.figure()
plt.plot(range(1, x_max), explained_variance_cumsum)
explained_variance_cumsum_fig.suptitle('Ratio of explained variance (cumsum)')
plt.xlabel('Rank')
plt.ylabel('Explained variance ratio')
filename = os.path.join(OUTPUT_DIR,
                        "explained_variance_cumsum.png")
plt.savefig(filename)
# Zoom in [1, N_COMP+1]
axes = explained_variance_cumsum_fig.axes
axes[0].set_xlim([1, N_COMP+1])
axes[0].set_ylim([explained_variance_cumsum[0], explained_variance_cumsum[N_COMP]+0.05])
plt.xticks(np.arange(1, N_COMP+1, 1.0))
filename = os.path.join(OUTPUT_DIR,
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
PC1_FR = X_proj_fr[:, 0]
PC2_FR = X_proj_fr[:, 1]
PC3_FR = X_proj_fr[:, 2]
PC1_GR = X_proj_ge[:, 0]
PC2_GR = X_proj_ge[:, 1]
PC3_GR = X_proj_ge[:, 2]
PC1 = np.append(PC1_FR, PC1_GR)
PC2 = np.append(PC2_FR, PC2_GR)
PC3 = np.append(PC3_FR, PC3_GR)
dataframe = pd.DataFrame({"SITE":SITE, "ID":ALL_SUBJECTS_ID, "PC1":PC1,
                          "PC2":PC2, "PC3":PC3})
dataframe.to_csv(OUTPUT_CSV)
