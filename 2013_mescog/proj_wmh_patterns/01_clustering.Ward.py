# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:22:38 2014

@author: lh239456

TODO:
 - how to determine and save cluster centers?

"""

import os
#import pickle

import numpy as np

import scipy
import scipy.cluster.hierarchy
import scipy.spatial.distance

import matplotlib.pyplot as plt

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "results", "wmh_patterns")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
#INPUT_SUBJECTS_DIR = os.path.join(INPUT_BASE_DIR,
#                                  "mescog", "datasets")
#INPUT_SUBJECTS = os.path.join(INPUT_SUBJECTS_DIR,
#                              "CAD-WMH-MNI-subjects.txt")
#INPUT_MASK = os.path.join(INPUT_DATASET_DIR, "wmh_mask.nii")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results", "wmh_patterns",
                          "clustering", "Ward")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_COMP_DIR_FMT = os.path.join(OUTPUT_DIR, "{i:03}")
OUTPUT_COND_DISTANCE_MATRIX = os.path.join(OUTPUT_DIR, "distance_matrix.cond.npy")
OUTPUT_DISTANCE_MATRIX = os.path.join(OUTPUT_DIR, "distance_matrix.full.npy")
OUTPUT_DISTANCE_MATRIX_IMG = os.path.join(OUTPUT_DIR, "distance_matrix.png")
OUTPUT_DISTANCE_MATRIX_BOXPLOT = os.path.join(OUTPUT_DIR, "distance_matrix.boxplot.png")
OUTPUT_LINKAGE = os.path.join(OUTPUT_DIR, "dendrogram.npy")
OUTPUT_FULL_DENDROGRAM = os.path.join(OUTPUT_DIR, "dendrogram.full.svg")
OUTPUT_LASTP_DENDROGRAM = os.path.join(OUTPUT_DIR, "dendrogram.lastp.svg")

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

# Distance matrix in condensed form (n*(n-1)/2 entries)
y = scipy.spatial.distance.pdist(X,
                                 metric='euclidean', p=2)
np.save(OUTPUT_COND_DISTANCE_MATRIX, y)

Y = scipy.spatial.distance.squareform(y)
np.save(OUTPUT_DISTANCE_MATRIX, Y)

distance_matrix_fig = plt.figure()
plt.matshow(Y, fignum=False)
plt.colorbar()
distance_matrix_fig.suptitle('Distance matrix')
distance_matrix_fig.savefig(OUTPUT_DISTANCE_MATRIX_IMG)

# Average distance to other points
av_dst = Y.mean(axis=0)
distance_matrix_boxplot = plt.figure()
plt.boxplot(av_dst)
plt.ylabel('Average distance')
distance_matrix_boxplot.suptitle('Average distance to other points')
distance_matrix_boxplot.savefig(OUTPUT_DISTANCE_MATRIX_BOXPLOT)

# Compute linkage
Z = scipy.cluster.hierarchy.linkage(y,
                                    method='single',
                                    metric='euclidean')
np.save(OUTPUT_LINKAGE, Z)

# Save dendrogram
dendrogram_full_fig = plt.figure()
R = scipy.cluster.hierarchy.dendrogram(Z,
           color_threshold=1,
           distance_sort='ascending')
dendrogram_full_fig.savefig(OUTPUT_FULL_DENDROGRAM)
dendrogram_lastp_fig = plt.figure()
R = scipy.cluster.hierarchy.dendrogram(Z,
           color_threshold=1,
           truncate_mode='lastp',
           distance_sort='ascending')
dendrogram_lastp_fig.savefig(OUTPUT_LASTP_DENDROGRAM)

# Cluster data
TS = []
for nb in n_clusters:
    T = scipy.cluster.hierarchy.fcluster(Z,
                                         criterion='maxclust',
                                         t=nb)
    TS.append(T)
    output_dir = OUTPUT_COMP_DIR_FMT.format(i=nb)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
#    print "Trying {nb} cluster(s)".format(nb=nb)
#
#    model.fit(X)
#    MODELS.append(model)
#    # Save model
#    filename = os.path.join(output_dir, "model.pkl")
#    with open(filename, "wb") as f:
#        pickle.dump(model, f)


#
#print "all files saved"
