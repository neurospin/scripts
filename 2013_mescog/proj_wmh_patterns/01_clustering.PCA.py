# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2013

@author: md238665

Use PCA to find the most important axes in data.

Use centered but not scaled data.

TODO:
 - kernel PCA?
 - interpretation?

"""

import os

import numpy as np

import sklearn
import sklearn.decomposition

import pandas as pd

import nibabel

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
INPUT_TRAIN_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")
INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

INPUT_CLINIC_DIR = os.path.join(INPUT_BASE_DIR,
                             "mescog", "proj_predict_cog_decline", "data")
INPUT_CSV = os.path.join(INPUT_CLINIC_DIR, "dataset_clinic_niglob_20140121.csv")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PCA_COMP = os.path.join(OUTPUT_DIR, "PCA.npy")
OUTPUT_PCA_PROJ = os.path.join(OUTPUT_DIR, "X_proj.npy")
OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_DIR, "{i:03}")
OUTPUT_MIN_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "min.{ID:04}.nii")
OUTPUT_MAX_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "max.{ID:04}.nii")
OUTPUT_PLOT_FMT = os.path.join(OUTPUT_DIR, "{feature}_pca{n_comp:02}")

##############
# Parameters #
##############

N_COMP = 10

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_TRAIN_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)
X_orig = np.load(INPUT_ORIG_DATASET)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

# Read subjects ID
with open(INPUT_TRAIN_SUBJECTS) as f:
    TRAIN_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Read  clinic data
clinical_data = pd.io.parsers.read_csv(INPUT_CSV, index_col=0)
csv_subjects_id = [int(subject_id[4:]) for subject_id in clinical_data.index]
clinical_data.index = csv_subjects_id

TRAIN_SUBJECTS_AGE = clinical_data["AGE_AT_INCLUSION"][TRAIN_SUBJECTS_ID]
TRAIN_SUBJECTS_VOLUME = clinical_data["BRAINVOL"][TRAIN_SUBJECTS_ID]
TRAIN_SUBJECTS_LL_COUNT = clinical_data["LLcount@M36"][TRAIN_SUBJECTS_ID]

# Get subjects without NAN in LL_COUNT
NAN_LL_COUNT = TRAIN_SUBJECTS_LL_COUNT.isnull()
NON_NAN_LL_COUNT = TRAIN_SUBJECTS_LL_COUNT[~NAN_LL_COUNT]
SUBJECT_ID_WITH_LL_COUNT = NON_NAN_LL_COUNT.index
SUBJECT_INDEX_WITH_LL_COUNT = np.where(~NAN_LL_COUNT)[0]
n_non_nan = SUBJECT_INDEX_WITH_LL_COUNT.shape[0]

# Compute decomposition
PCA = sklearn.decomposition.PCA()
PCA.fit(X)

# Store components
np.save(OUTPUT_PCA_COMP, PCA.components_)

# Project subjects onto PC & save it
# scikit-learn projects on min(n_samples, n_features)
X_proj = PCA.transform(X)

np.save(OUTPUT_PCA_PROJ, X_proj)

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
    extremum_sub = (min_sub, max_sub) = (X_proj[:, i].argmin(), X_proj[:, i].argmax())
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

import matplotlib.pyplot as plt
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

# For each component of the PCA, we plot the distribution of age
# and brain volume
for i in range(N_COMP):
    # Age distribution
    pca_age_fig = plt.figure()
    plt.scatter(X_proj[:, i], TRAIN_SUBJECTS_AGE)
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
    plt.scatter(X_proj[:, i], TRAIN_SUBJECTS_VOLUME)
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
plt.scatter(X_proj[:, 1], X_proj[:, 2])
plt.xlim([-100, 40])
plt.ylim([-60, 40])
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
NAME = os.path.join(OUTPUT_DIR, "pca3_pca2")
plt.savefig(NAME)

# Plot of 3rd component of PCA along 2nd component
# with colors so subjects without LL_COUNT don't appear
colors = TRAIN_SUBJECTS_LL_COUNT
pca2_pca3_fig_colors = plt.figure()
c = plt.scatter(X_proj[:, 1], X_proj[:, 2], c=colors, s=50)
c.set_alpha(0.75)
plt.xlim([-300, 125])
plt.ylim([-150, 125])
plt.colorbar()
pca2_pca3_fig_colors.suptitle("Position of subjects along 2nd and 3rd "
                              "principal components (n={n})".format(n=n_non_nan))
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
plt.text(x=100, y=5, s='PCA_2', horizontalalignment='center',
         verticalalignment='center')
plt.text(x=0, y=100, s='PCA_3', horizontalalignment='left',
         verticalalignment='bottom')
for i, subject_id in zip(SUBJECT_INDEX_WITH_LL_COUNT, SUBJECT_ID_WITH_LL_COUNT):
    plt.annotate(str(subject_id), xy=(X_proj[i, 1], X_proj[i, 2]))
NAME = os.path.join(OUTPUT_DIR, "pca3_pca2_colors_test")
plt.savefig(NAME)
print "position graphs saved"
