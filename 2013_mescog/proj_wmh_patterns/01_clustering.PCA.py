# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2013

@author: md238665

Use PCA to find the most important axes in data.

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

INPUT_BASE_DIR = "/volatile/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "results")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
INPUT_SUBJECTS_DIR = os.path.join(INPUT_BASE_DIR,
                                  "mescog", "datasets")
INPUT_SUBJECTS = os.path.join(INPUT_SUBJECTS_DIR,
                              "CAD-WMH-MNI-subjects.txt")
INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/volatile/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results", "PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_PCA = os.path.join(OUTPUT_DIR, "PCA.pkl")
OUTPUT_COMP_IMAGE_FMT = os.path.join(OUTPUT_DIR, "{i:03}.nii")

##############
# Parameters #
##############

N_COMP = 10

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
print "Data loaded: {s[0]}x{s[1]}".format(s=X.shape)

# Compute decomposition
PCA = sklearn.decomposition.PCA()
PCA.fit(X)
with open(OUTPUT_PCA, "wb") as f:
    pickle.dump(PCA, f)

# Save the components into brain-like images
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0
for i in range(N_COMP):
    comp_data = np.zeros(binary_mask.shape)
    comp_data[binary_mask] = PCA.components_[i, :]
    comp_im = nibabel.Nifti1Image(comp_data, babel_mask.get_affine())
    name = OUTPUT_COMP_IMAGE_FMT.format(i=i)
    nibabel.save(comp_im, name)
