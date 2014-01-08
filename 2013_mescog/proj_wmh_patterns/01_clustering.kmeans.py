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

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/"
INPUT_DATASET_DIR = os.path.join(INPUT_BASE_DIR,
                                 "mescog", "results", "wmh_patterns")
INPUT_DATASET = os.path.join(INPUT_DATASET_DIR,
                             "train.std.npy")
INPUT_SUBJECTS_DIR = os.path.join(INPUT_BASE_DIR,
                                  "mescog", "datasets")
INPUT_SUBJECTS = os.path.join(INPUT_SUBJECTS_DIR,
                              "CAD-WMH-MNI-subjects.txt")

OUTPUT_BASE_DIR = "/neurospin/"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "mescog", "results", "wmh_patterns",
                          "clustering", "kmeans")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

##############
# Parameters #
##############

K = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#################
# Actual script #
#################

# Read input data
X = np.load(INPUT_DATASET)
n, p = s = X.shape
print "Data loaded {s}".format(s=s)

MODELS=[]
MODEL_FILENAMES=[]
for k in K:
    print "Trying k={k}".format(k=k)
    model = sklearn.cluster.KMeans(n_clusters=k,
                                   init='k-means++',
                                   n_init=10,
                                   n_jobs=1)
    model.fit(X)
    filename = os.path.join(OUTPUT_DIR, str(k)+".pkl")
    MODEL_FILENAMES.append(filename)
    MODELS.append(model)
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Post-processing
INERTIA=np.zeros((len(K), 2))
BIC=np.zeros((len(K), 2))
for i, (k, filename) in enumerate(zip(K, MODEL_FILENAMES)):
    with open(filename) as f:
        model = pickle.load(f)
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
plt.bar(K, BIC[:, 1])
bic_fig.suptitle('BIC')
plt.xlabel('# of clusters')
plt.ylabel('BIC')
filename=os.path.join(OUTPUT_DIR,
                      "BIC.png")
plt.savefig(filename)

inertia_fig=plt.figure()
plt.bar(K, INERTIA[:, 1])
inertia_fig.suptitle('Inertia')
plt.xlabel('# of clusters')
plt.ylabel('Inertia')
filename=os.path.join(OUTPUT_DIR,
                      "Inertia.png")
plt.savefig(filename)
#plt.show()
