# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:07:23 2014

@author: md238665

Create datasets from the cropped DB:
 - convert to grayscale
 - subsample
 - center (locally and globally as in Olivetti faces)

"""


import os
import json

import numpy as np
import scipy.misc
import scipy.ndimage
import skimage
import skimage.color
import skimage.transform
import pandas as pd
import matplotlib
import matplotlib.pylab as plt

##################
# Input & output #
##################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/AR_faces"

INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "raw_data",
                         "cropped_faces")
INPUT_POPULATION = os.path.join(INPUT_DIR,
                                "population.csv")

OUTPUT_DIR = os.path.join(INPUT_BASE_DIR,
                          "dataset")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_DATASET = os.path.join(OUTPUT_DIR,
                             "X.npy")
OUTPUT_VAR_FILE = os.path.join(OUTPUT_DIR,
                               "pixel_var.png")

##############
# Parameters #
##############

# Real shape of images
INPUT_ORIG_SHAPE = (165, 120)

# Resize
INPUT_RESIZE_SHAPE = (38, 27)


#############
# Functions #
#############


#################
# Actual script #
#################

if __name__ == "__main__":
    # Read population
    population = pd.io.parsers.read_csv(INPUT_POPULATION)
    n = len(population)

    # Read images, convert to gray, subsample and store
    images = np.empty((n, np.prod(INPUT_RESIZE_SHAPE)))
    for i, image_filename in enumerate(population['file']):
        # Read image
        rgb_data = scipy.misc.imread(image_filename)
        # Gray scale
        float_data = skimage.color.rgb2gray(rgb_data)
        # Rescaling
        rescaled_data = skimage.transform.resize(float_data,
                                                 INPUT_RESIZE_SHAPE)
        images[i] = rescaled_data.ravel()

    # Globale centering
    global_centered_images = images - images.mean(axis=0)

    # Local centering
    local_centering = global_centered_images.mean(axis=1).reshape(n, -1)
    local_centered_images = global_centered_images - local_centering

    # Save
    np.save(OUTPUT_DATASET, local_centered_images)

    # Pixel-wise variance
    var = local_centered_images.var(axis=0).reshape(INPUT_RESIZE_SHAPE)
    fig = plt.figure()
    my_cmap = matplotlib.cm.get_cmap('gray')
    im = plt.imshow(var,
                cmap=my_cmap)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(OUTPUT_VAR_FILE)
    fig.show()
