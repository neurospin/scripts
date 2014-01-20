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
import pickle

import numpy as np

import sklearn
import sklearn.decomposition

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

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PCA = os.path.join(OUTPUT_DIR, "PCA.pkl")
OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_DIR, "{i:03}")
OUTPUT_MIN_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "min.{ID:04}.nii")
OUTPUT_MAX_SUBJECT_FMT = os.path.join(OUTPUT_COMP_DIR_FMT,
                                      "max.{ID:04}.nii")

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

# Compute decomposition
PCA = sklearn.decomposition.PCA()
PCA.fit(X)
with open(OUTPUT_PCA, "wb") as f:
    pickle.dump(PCA, f)

# Project subjects onto dimensions
# scikit-learn projects on min(n_samples, n_features)
X_proj = PCA.transform(X)

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

# Plot percentage of explained variance
explained_variance = PCA.explained_variance_ratio_
explained_variance_cumsum = explained_variance.cumsum()

import matplotlib.pyplot as plt
explained_variance_fig = plt.figure()
plt.plot(explained_variance)
explained_variance_fig.suptitle('Ratio of explained variance')
plt.xlabel('Rank')
plt.ylabel('Explained variance ratio')
filename=os.path.join(OUTPUT_DIR,
                      "explained_variance.png")
plt.savefig(filename)
# Zoom in [0, N_COMP]
axes = explained_variance_fig.axes
axes[0].set_xlim([0, N_COMP])
axes[0].set_ylim([explained_variance[N_COMP], explained_variance[0]])
filename=os.path.join(OUTPUT_DIR,
                      "explained_variance.zoom.png")
plt.savefig(filename)

explained_variance_cumsum_fig = plt.figure()
plt.plot(explained_variance_cumsum)
explained_variance_cumsum_fig.suptitle('Ratio of explained variance (cumsum)')
plt.xlabel('Rank')
plt.ylabel('Explained variance ratio')
filename=os.path.join(OUTPUT_DIR,
                      "explained_variance_cumsum.png")
plt.savefig(filename)
# Zoom in [0, N_COMP]
axes = explained_variance_cumsum_fig.axes
axes[0].set_xlim([0, N_COMP])
axes[0].set_ylim([explained_variance_cumsum[0], explained_variance_cumsum[N_COMP]])
filename=os.path.join(OUTPUT_DIR,
                      "explained_variance_cumsum.zoom.png")
plt.savefig(filename)
