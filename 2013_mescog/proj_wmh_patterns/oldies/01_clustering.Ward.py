# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:22:38 2014

@author: lh239456

Use hieararchical clustering to identify patterns.

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
INPUT_TRAIN_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "french.npy")
INPUT_TRAIN_SUBJECTS = os.path.join(INPUT_DATASET_DIR,
                                    "french-subjects.txt")
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
OUTPUT_CENTER_FMT = os.path.join(OUTPUT_DIR_FMT, "{i:02}.mean.nii")
OUTPUT_CLOSEST_SUBJECT_FMT = os.path.join(OUTPUT_DIR_FMT,
                                          "{i:02}.nearest.{ID:04}.nii")

##############
# Parameters #
##############

n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_TRAIN_DATASET)
n, p = s = X.shape
print "Data loaded {s}".format(s=s)

# Read mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data()
binary_mask = mask != 0

# Read subjects ID
with open(INPUT_TRAIN_SUBJECTS) as f:
    TRAIN_SUBJECTS_ID = np.array([int(l) for l in f.readlines()])

# Compute distance matrix
y = scipy.spatial.distance.pdist(X, metric='euclidean', p=2) # condensed (n*(n-1)/2 entries)
Y = scipy.spatial.distance.squareform(y) # full form
np.save(OUTPUT_DISTANCE_MATRIX, Y)

# Plot it
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
    # Assignment
    T = scipy.cluster.hierarchy.fcluster(Z,
                                         criterion='maxclust',
                                         t=nb)
    # The fcluster function numbers cluster from 1 to nb.
    # For consistency with other scripts we number them from 0 to nb-1.
    T = T - 1
    TS.append(T)

    # Save model & assignment
    output_dir = OUTPUT_DIR_FMT.format(k=nb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assign_filename = os.path.join(output_dir, "assign.npy")
    np.save(assign_filename, T)
    # Find cluster centers
    centers = np.zeros((nb, p))
    for i in range(nb):
        index = np.where(T == i)[0]
        # Compute center
        all_data = X[index]
        mean_data = all_data.mean(axis=0)
        centers[i] = mean_data
        # Create image of center and save it
        im_data = np.zeros(mask.shape)
        im_data[binary_mask] = mean_data
        im = nibabel.Nifti1Image(im_data, babel_mask.get_affine())
        filename = OUTPUT_CENTER_FMT.format(k=nb, i=i)
        nibabel.save(im, filename)
    # Save the centers
    centers_filename = os.path.join(output_dir, "centers.npy")
    np.save(centers_filename, centers)
    # Compute distance to all center (as in kmeans)
    dst_to_centers = scipy.spatial.distance.cdist(X, centers)
    dst_to_centers_filename = os.path.join(output_dir, "dst_to_centers.npy")
    np.save(dst_to_centers_filename, dst_to_centers)
    # Store number of members in each cluster and closest to center subject
    n_points_per_cluster_filename = os.path.join(output_dir, "n_points_per_cluster.txt")
    f = open(n_points_per_cluster_filename, "w+")
    for i in range(nb):
        index = np.where(T == i)[0]
        print >> f, index.shape[0]
        # Compute the distance to this center
        dst_to_this_center = dst_to_centers[:, i]
        # Closest subject index (in X)
        closest_subject_index = np.argmin(dst_to_this_center)
#        if T[closest_subject_index != i]:
#            print "Dudu screw it up"
        closest_subject_ID = TRAIN_SUBJECTS_ID[closest_subject_index]
        # Convert to image
        im_data = np.zeros(mask.shape)
        im_data[binary_mask] = X[closest_subject_index, :]
        im = nibabel.Nifti1Image(im_data, babel_mask.get_affine())
        name = OUTPUT_CLOSEST_SUBJECT_FMT.format(k=nb, i=i,
                                                 ID=closest_subject_ID)
        nibabel.save(im, name)
    f.close()
