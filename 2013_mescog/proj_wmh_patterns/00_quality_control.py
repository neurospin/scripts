# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:41:58 2014

@author: lh239456

Compute the average image of all the dataset.

"""

import os

import numpy as np

import nibabel

##################
# Input & output #
##################


INPUT_BASE_DIR = "/neurospin/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "mescog", "datasets")
INPUT_DATASET = os.path.join(INPUT_DIR,
                             "CAD-WMH-MNI.npy")
INPUT_MASK = os.path.join(INPUT_DIR,
                          "MNI152_T1_2mm_brain_mask.nii.gz")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results",
                          "wmh_patterns", "quality_control")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_AVERAGE_BRAIN = os.path.join(OUTPUT_DIR, "average_brain.nii")

##############
# Parameters #
##############

IM_SHAPE = (91, 109, 91)

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
ORIG_SHAPE = X.shape
print "Loaded images dataset {s}".format(s=ORIG_SHAPE)

# Create average brain
X_average = np.mean(X, axis=0)
print "X_average shape : {s}".format(s=X_average.shape)

# Reshape brain image
X_average.shape = IM_SHAPE
print "Reshaped X_average shape : {s}".format(s=X_average.shape)

# Save the mean into brain-like images
babel_mask = nibabel.load(INPUT_MASK)
average_brain = nibabel.Nifti1Image(X_average, babel_mask.get_affine())
nibabel.save(average_brain, OUTPUT_AVERAGE_BRAIN)
print "Average brain saved"

# Compute some statistics on data
import matplotlib.pyplot as plt

step = 0.2

x_zero_hist = plt.figure()
range_zero = np.arange(0, X.max() + step, step)
plt.hist(X.flatten(), bins=range_zero)
x_zero_hist.suptitle("histogram of all subjects including zero")
plt.xlabel('value')
plt.ylabel('nb of occurences')
filename = os.path.join(OUTPUT_DIR, "x_zero_hist.png")
plt.savefig(filename)

x_non_zero_hist = plt.figure()
range_non_zero = np.arange(step, X.max() + step, step)
plt.hist(X.flatten(), bins=range_non_zero)
x_non_zero_hist.suptitle("histogram of all subjects excluding zero")
plt.xlabel('value')
plt.ylabel('nb of occurences')
filename = os.path.join(OUTPUT_DIR, "x_non_zero_hist.png")
plt.savefig(filename)

x_average_hist = plt.figure()
range_zero = np.arange(0, X_average.max() + step, step)
plt.hist(X_average.flatten(), bins=range_zero)
x_average_hist.suptitle("histogram of average subject including zero")
plt.xlabel('value')
plt.ylabel('nb of occurences')
filename = os.path.join(OUTPUT_DIR, "x_average_zero_hist.png")
plt.savefig(filename)

x_average_non_zero_hist = plt.figure()
range_non_zero = np.arange(step, X_average.max() + step, step)
plt.hist(X_average.flatten(), bins=range_non_zero)
x_average_non_zero_hist.suptitle("histogram of average subject excluding zero")
plt.xlabel('value')
plt.ylabel('nb of occurences')
filename = os.path.join(OUTPUT_DIR, "x_average_non_zero_hist.png")
plt.savefig(filename)

print "Histograms saved"
