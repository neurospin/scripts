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
                             "CAD-WMH-MNI.npy")

INPUT_RESOURCES_DIR = os.path.join(INPUT_BASE_DIR,
                                   "mescog", "neuroimaging", "ressources")
INPUT_MASK = os.path.join(INPUT_RESOURCES_DIR,
                          "MNI152_T1_1mm_brain_mask.nii.gz")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns", "quality_control")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_AVERAGE_BRAIN = os.path.join(OUTPUT_DIR, "average_brain.nii")

OUTPUT_X_STATISTICS = os.path.join(OUTPUT_DIR,
                                   "X.statistics.csv")
OUTPUT_X_AVERAGE_STATISTICS = os.path.join(OUTPUT_DIR,
                                           "X_average.statistics.csv")

OUTPUT_HIST_AVERAGE_ZERO = os.path.join(OUTPUT_DIR,
                                        "X_average.hist.png")
OUTPUT_HIST_AVERAGE_NON_ZERO = os.path.join(OUTPUT_DIR,
                                            "X_average.non_zero.hist.png")
OUTPUT_HIST_N_VOX_NON_ZERO = os.path.join(OUTPUT_DIR,
                                          "n_non_zero.hist.png")

##############
# Parameters #
##############

IM_SHAPE = (182, 218, 182)
HISTO_N_BINS = 10
HIST_THRESHOLD = 0.02

#################
# Actual script #
#################

# Read masked data
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

# Save the mean into brain-like images
data = np.zeros(mask.shape)
data[binary_mask] = X_average
average_brain = nibabel.Nifti1Image(data, babel_mask.get_affine())
nibabel.save(average_brain, OUTPUT_AVERAGE_BRAIN)
print "Average brain saved"

# Compute some statistics on masked data
X_average_mask_desc = pd.DataFrame(X_average).describe()
X_average_mask_desc.to_csv(OUTPUT_X_AVERAGE_STATISTICS, header=False, sep=':')

# X is large so we trick pandas a bit here
X_desc = X_average_mask_desc.copy()
X.shape = (np.prod(ORIG_SHAPE), )
X_desc[0]["count"] = np.prod(X.shape)
X_desc[0]["mean"] = X.mean()
X_desc[0]["std"] = X.std()
X_desc[0]["min"] = X.min()
X_desc[0]["max"] = X.max()
quartiles = np.percentile(X, [0.25, 0.5, 0.75])
X_desc[0]["25%"] = quartiles[0]
X_desc[0]["50%"] = quartiles[1]
X_desc[0]["75%"] = quartiles[2]
X_desc.to_csv(OUTPUT_X_STATISTICS, header=False, sep=':')
X.shape = ORIG_SHAPE

n_vox_non_zero = X.sum(axis=1)

# Plot some histograms
import matplotlib.pyplot as plt

# Histogram of masked average subject, including 0
x_average_hist = plt.figure()
(count, bins, _) = plt.hist(X_average.ravel(), bins=HISTO_N_BINS)
x_average_hist.suptitle("Histogram of average subject")
plt.xlabel('Value')
plt.ylabel('# of occurences')
plt.savefig(OUTPUT_HIST_AVERAGE_ZERO)

# Histogram of masked average subject, excluding below HIST_THRESHOLD
x_average_non_zero_hist = plt.figure()
data = X_average[X_average >= HIST_THRESHOLD]
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
