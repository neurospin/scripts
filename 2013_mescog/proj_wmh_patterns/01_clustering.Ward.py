# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:22:38 2014

@author: lh239456

TODO:
 - how to determine and save cluster centers?

"""

import os

import numpy as np

import scipy
import scipy.cluster.hierarchy
import scipy.spatial.distance

import matplotlib.pyplot as plt

import nibabel

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "proj_wmh_patterns")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
INPUT_TRAIN_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                    "train_subjects.txt")
INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "proj_wmh_patterns",
                          "clustering", "Ward")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DISTANCE_MATRIX = os.path.join(OUTPUT_DIR, "distance_matrix.npy")
OUTPUT_DISTANCE_MATRIX_IMG = os.path.join(OUTPUT_DIR, "distance_matrix.png")
OUTPUT_DISTANCE_MATRIX_BOXPLOT = os.path.join(OUTPUT_DIR,
                                              "distance_matrix.boxplot.png")
OUTPUT_LINKAGE = os.path.join(OUTPUT_DIR, "dendrogram.npy")
OUTPUT_FULL_DENDROGRAM = os.path.join(OUTPUT_DIR, "dendrogram.full.svg")
OUTPUT_LASTP_DENDROGRAM = os.path.join(OUTPUT_DIR, "dendrogram.lastp.svg")

OUTPUT_DIR_FMT = os.path.join(OUTPUT_DIR, "{k:02}")
OUTPUT_CENTER_FMT = os.path.join(OUTPUT_DIR_FMT,
                                 "{i:02}.nii")

##############
# Parameters #
##############

n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
n, p = s = X.shape
print "Data loaded {s}".format(s=s)

# Read subjects ID
with open(INPUT_TRAIN_SUBJECTS) as f:
    TRAIN_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

# Distance matrix in condensed form (n*(n-1)/2 entries)
y = scipy.spatial.distance.pdist(X,
                                 metric='euclidean', p=2)

Y = scipy.spatial.distance.squareform(y)
np.save(OUTPUT_DISTANCE_MATRIX, Y)

distance_matrix_fig = plt.figure()
plt.matshow(Y, fignum=False)
plt.colorbar()
distance_matrix_fig.suptitle('Distance matrix')
distance_matrix_fig.savefig(OUTPUT_DISTANCE_MATRIX_IMG)

# Compute linkage
Z = scipy.cluster.hierarchy.linkage(X,
                                    method='ward',
                                    metric='euclidean')
np.save(OUTPUT_LINKAGE, Z)

# Save dendrogram
dendrogram_full_fig = plt.figure()
R = scipy.cluster.hierarchy.dendrogram(Z,
                                       color_threshold=1,
                                       distance_sort='ascending',
                                       labels=TRAIN_SUBJECTS_ID)
dendrogram_full_fig.savefig(OUTPUT_FULL_DENDROGRAM)
dendrogram_lastp_fig = plt.figure()
R = scipy.cluster.hierarchy.dendrogram(Z,
                                       color_threshold=1,
                                       truncate_mode='lastp',
                                       distance_sort='ascending',
                                       labels=TRAIN_SUBJECTS_ID)
dendrogram_lastp_fig.savefig(OUTPUT_LASTP_DENDROGRAM)

# Cluster data
TS = []
for nb in n_clusters:
    print "Trying {nb} cluster(s)".format(nb=nb)

    T = scipy.cluster.hierarchy.fcluster(Z,
                                         criterion='maxclust',
                                         t=nb)
    TS.append(T)

    # Save model
    output_dir = OUTPUT_DIR_FMT.format(k=nb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, "model.npy")
    np.save(filename, T)
    # Create an average brain image
    for i in range(1, nb+1):
        index = np.where(T == i)[0]
        all_data = X[index]
        mean_data = all_data.mean(axis=0)
        # Create image and save it
        im_data = np.zeros(mask.shape)
        im_data[binary_mask] = mean_data
        im = nibabel.Nifti1Image(im_data, babel_mask.get_affine())
        filename = OUTPUT_CENTER_FMT.format(k=nb, i=i)
        nibabel.save(im, filename)
