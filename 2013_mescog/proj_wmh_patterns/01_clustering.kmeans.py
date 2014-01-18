# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:42:39 2013

@author: md238665

Use K-means clustering to identify patterns.

"""

import os
import pickle

import numpy as np

import sklearn
import sklearn.cluster

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
                          "clustering", "kmeans")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DIR_FMT = os.path.join(OUTPUT_DIR, "{k:02}")
OUTPUT_CENTER_FMT = os.path.join(OUTPUT_DIR_FMT, "{i:02}.mean.nii")
OUTPUT_CLOSEST_SUBJECT_FMT = os.path.join(OUTPUT_DIR_FMT,
                                          "{i:02}.nearest.{ID:03}.nii")

##############
# Parameters #
##############

K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

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

# Fitting
# We use fit_transform to return the distance to all centers
# and then predict
MODELS=[]
for k in K:
    print "Trying k={k}".format(k=k)
    model = sklearn.cluster.KMeans(n_clusters=k,
                                   init='k-means++',
                                   n_init=10,
                                   n_jobs=1)
    # Fit and return distance to each center
    dst_to_centers = model.fit_transform(X)
    MODELS.append(model)
    # Assignement
    assign = model.predict(X)
    # Store model, distance to center & assignment
    output_dir = OUTPUT_DIR_FMT.format(k=k)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_filename = os.path.join(output_dir,
                                 "model.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    dst_to_centers_filename = os.path.join(output_dir, "dst_to_centers.npy")
    np.save(dst_to_centers_filename, dst_to_centers)
    assign_filename = os.path.join(output_dir, "assign.npy")
    np.save(assign_filename, assign)
    # Store centers as images, closest-to-center subject
    # and umber of members in each cluster.
    n_points_per_cluster_filename = os.path.join(output_dir, "n_points_per_cluster.txt")
    f = open(n_points_per_cluster_filename, "w+")
    for i in range(k):
        index = np.where(assign == i)[0]
        print >> f, index.shape[0]
        # Distance to this center for each point
        dst_to_this_center = dst_to_centers[:, i]
        # Closest subject index (in X)
        closest_subject_index = np.argmin(dst_to_this_center)
        closest_subject_ID = TRAIN_SUBJECTS_ID[closest_subject_index]
        # Convert to image
        im_data = np.zeros(mask.shape)
        im_data[binary_mask] = X[closest_subject_index, :]
        im = nibabel.Nifti1Image(im_data, babel_mask.get_affine())
        name = OUTPUT_CLOSEST_SUBJECT_FMT.format(k=k, i=i,
                                                 ID=closest_subject_ID)
        nibabel.save(im, name)
        # Create image of center and save it
        im_data = np.zeros(mask.shape)
        im_data[binary_mask] = model.cluster_centers_[i, :]
        im = nibabel.Nifti1Image(im_data, babel_mask.get_affine())
        name = OUTPUT_CENTER_FMT.format(k=k, i=i)
        nibabel.save(im, name)
    f.close()

# Post-processing
INERTIA=np.zeros((len(K), 2))
BIC=np.zeros((len(K), 2))
for i, (k, model) in enumerate(zip(K, MODELS)):
    INERTIA[i, 0] = k
    INERTIA[i, 1] = model.inertia_
    # Compute the BIC (http://stackoverflow.com/questions/15839774/how-to-calculate-bic-for-k-means-clustering-in-r)
    bic = model.inertia_ + .5*k*p*np.log(n)
    BIC[i, 0] = k
    BIC[i, 1] = bic

INERTIA_filename = os.path.join(OUTPUT_DIR, "Inertia.txt")
np.savetxt(INERTIA_filename, INERTIA)
BIC_filename = os.path.join(OUTPUT_DIR, "BIC.txt")
np.savetxt(BIC_filename, BIC)

import matplotlib.pyplot as plt

bic_fig = plt.figure()
plt.plot(K, BIC[:, 1])
bic_fig.suptitle('BIC')
plt.xlabel('# of clusters')
plt.ylabel('BIC')
filename=os.path.join(OUTPUT_DIR,
                      "BIC.png")
plt.savefig(filename)

inertia_fig=plt.figure()
plt.plot(K, INERTIA[:, 1])
inertia_fig.suptitle('Inertia')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
filename=os.path.join(OUTPUT_DIR,
                      "Inertia.png")
plt.savefig(filename)
#plt.show()
