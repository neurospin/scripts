# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 14:16:58 2014

@author: fh235918

Dump the Olivetti dataset. Data are centered both globally and locally.

Warning: we don't shuffle the data.

"""

import os

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

OUTPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/Olivetti_faces"
OUTPUT_DATASET_FILE = os.path.join(OUTPUT_BASE_DIR,
                                   "X.npy")
OUTPUT_TARGET_FILE = os.path.join(OUTPUT_BASE_DIR,
                                  "y.npy")

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces()
faces = dataset.data

n, p = shape = faces.shape

# global centering
faces_centered_global = faces - faces.mean(axis=0)

# local centering
local_centering = faces_centered_global.mean(axis=1).reshape(n, -1)
faces_centered_local = faces_centered_global - local_centering

print("Dataset shape: {s}".format(s=shape))

# Load ground truth (just in case)
y = dataset.target

###############################################################################
# Dump it
np.save(OUTPUT_DATASET_FILE, faces_centered_local)
np.save(OUTPUT_TARGET_FILE, y)

###############################################################################
# Plot an original face and the centered versions
import matplotlib
import matplotlib.pylab as plt
DEMO_FACE = 0
im1 = faces[0, :].reshape(64, 64)
im1_bounds = [im1.min(), im1.max()]
im2 = faces_centered_global[0, :].reshape(64, 64)
im2_bounds = [im2.min(), im2.max()]
im3 = faces_centered_local[0, :].reshape(64, 64)
im3_bounds = [im3.min(), im3.max()]

bounds = [min([im1_bounds[0], im2_bounds[0], im3_bounds[0]]),
          max([im1_bounds[1], im2_bounds[1], im3_bounds[1]])]

fig = plt.figure()
my_cmap = matplotlib.cm.get_cmap('gray')
plt.subplot(1, 3, 1)
im = plt.imshow(im1,
                vmin=bounds[0], vmax=bounds[1],
                cmap=my_cmap)
plt.title('Original')
plt.subplot(1, 3, 2)
plt.imshow(im2,
           vmin=bounds[0], vmax=bounds[1],
           cmap=my_cmap)
plt.title('Global centering')
plt.subplot(1, 3, 3)
plt.imshow(im3,
           vmin=bounds[0], vmax=bounds[1],
           cmap=my_cmap)
plt.title('Local centering')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.show()
