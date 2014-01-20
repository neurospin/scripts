# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:41:58 2014

@author: lh239456

Compute some statistics on data:
 - compute the average image (average accross subjects)
 - describe dataset and average image (main statistics)
 - histogram of average image (inside the MNI mask)
 - histogram of number of voxel showing WMH

"""

import os

import numpy as np

import pandas as pd

import nibabel

##################
# Input & output #
##################


INPUT_BASE_DIR = "/neurospin/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "mescog", "datasets")
INPUT_DATASET = os.path.join(INPUT_DIR,
                             "CAD-WMH-MNI.without_outliers.npy")
INPUT_MASK = os.path.join(INPUT_DIR,
                          "MNI152_T1_2mm_brain_mask.nii.gz")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "quality_control")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_AVERAGE_BRAIN = os.path.join(OUTPUT_DIR, "average_brain.nii")

OUTPUT_X_MASK_STATISTICS = os.path.join(OUTPUT_DIR,
                                        "X_mask.statistics.csv")
OUTPUT_X_AVERAGE_MASK_STATISTICS = os.path.join(OUTPUT_DIR,
                                                "X_average_mask.statistics.csv")

OUTPUT_HIST_AVERAGE_ZERO = os.path.join(OUTPUT_DIR,
                                        "X_average_mask.hist.png")
OUTPUT_HIST_AVERAGE_NON_ZERO = os.path.join(OUTPUT_DIR,
                                            "X_average_mask.non_zero.hist.png")
OUTPUT_HIST_N_VOX_NON_ZERO = os.path.join(OUTPUT_DIR,
                                          "n_non_zero.hist.png")

##############
# Parameters #
##############

IM_SHAPE = (91, 109, 91)
HISTO_N_BINS = 10
HIST_THRESHOLD = 0.02

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
ORIG_SHAPE = X.shape
print "Loaded images dataset {s}".format(s=ORIG_SHAPE)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0
mask_lin = mask.ravel()
mask_index = np.where(mask_lin)[0]

# Create average brain
X_average = np.mean(X, axis=0)
print "X_average shape : {s}".format(s=X_average.shape)

# Reshape brain image
X_average.shape = IM_SHAPE
print "Reshaped X_average shape : {s}".format(s=X_average.shape)

# Save the mean into brain-like images
average_brain = nibabel.Nifti1Image(X_average, babel_mask.get_affine())
nibabel.save(average_brain, OUTPUT_AVERAGE_BRAIN)
print "Average brain saved"

# Mask data
X_mask = X[:, mask_index]
X_average_mask = X_average[binary_mask]

# Compute some statistics on masked data
X_mask_desc = pd.DataFrame(X_mask.ravel()).describe()
X_mask_desc.to_csv(OUTPUT_X_MASK_STATISTICS, header=False, sep=':')

X_average_mask_desc = pd.DataFrame(X_average_mask).describe()
X_average_mask_desc.to_csv(OUTPUT_X_AVERAGE_MASK_STATISTICS, header=False, sep=':')

n_vox_non_zero = X_mask.sum(axis=1)

# Plot some histograms
import matplotlib.pyplot as plt

# Histogram of masked average subject, including 0
x_average_hist = plt.figure()
(count, bins, _) = plt.hist(X_average_mask.ravel(), bins=HISTO_N_BINS)
x_average_hist.suptitle("Histogram of average subject")
plt.xlabel('Value')
plt.ylabel('# of occurences')
plt.savefig(OUTPUT_HIST_AVERAGE_ZERO)

# Histogram of masked average subject, excluding below HIST_THRESHOLD
x_average_non_zero_hist = plt.figure()
data = X_average_mask[X_average_mask >= HIST_THRESHOLD]
plt.hist(data, bins=bins)
x_average_non_zero_hist.suptitle("Histogram of average subject (>{thresh})"
                                 .format(thresh=HIST_THRESHOLD))
plt.xlabel('Value')
plt.ylabel('# of occurences')
plt.savefig(OUTPUT_HIST_AVERAGE_NON_ZERO)

# Histogram number of voxels
n_vox_non_zero_hist = plt.figure()
plt.hist(n_vox_non_zero, bins=10)
n_vox_non_zero_hist.suptitle("Histogram of the number of non-zero voxels per-subject")
plt.xlabel('Value')
plt.ylabel('# of occurences')
plt.savefig(OUTPUT_HIST_N_VOX_NON_ZERO)
