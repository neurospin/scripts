# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:07:23 2014

@author: md238665

Create datasets from the cropped DB:
 - whole DB
 - only non-occluded images
 - non-occluded and neutral lighting images

Preprocessings:
 - convert to grayscale
 - subsample

Each dataset is centered (locally and globally as in Olivetti faces).

"""


import os

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
OUTPUT_WHOLE_DATASET = os.path.join(OUTPUT_DIR,
                                    "X.npy")
OUTPUT_VAR_FILE = os.path.join(OUTPUT_DIR,
                               "pixel_var.png")

OUTPUT_NO_DATASET = os.path.join(OUTPUT_DIR,
                                 "X.non_occluded.npy")
OUTPUT_NO_MASK = os.path.join(OUTPUT_DIR,
                              "non_occluded.npy")
OUTPUT_NO_VAR_FILE = os.path.join(OUTPUT_DIR,
                                  "non_occluded_pixel_var.png")

OUTPUT_NONL_DATASET = os.path.join(OUTPUT_DIR,
                                   "X.nonl.npy")
OUTPUT_NONL_MASK = os.path.join(OUTPUT_DIR,
                                "nonl.npy")
OUTPUT_NONL_VAR_FILE = os.path.join(OUTPUT_DIR,
                                    "nonl_pixel_var.png")

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

def global_local_centering(images):
    n = images.shape[0]

    # Global centering
    global_centering = images.mean(axis=0)
    global_centered_images = images - global_centering

    # Local centering
    local_centering = global_centered_images.mean(axis=1).reshape(n, -1)
    local_centered_images = global_centered_images - local_centering
    return (local_centered_images, global_centering, local_centering)


def save_pixel_var(data, output_file):
    var = data.var(axis=0).reshape(INPUT_RESIZE_SHAPE)
    fig = plt.figure()
    my_cmap = matplotlib.cm.get_cmap('gray')
    im = plt.imshow(var,
                    cmap=my_cmap)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig(output_file)
    fig.show()

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

    # First dataset: whole population
    local_centered_images, _, _ = global_local_centering(images)
    np.save(OUTPUT_WHOLE_DATASET, local_centered_images)
    save_pixel_var(local_centered_images, OUTPUT_VAR_FILE)

    # Second dataset: non-occluded faces (no)
    non_occluded = population['occlusion'].isnull().as_matrix()
    no_images = local_centered_images[non_occluded]
    no_local_centered_images, _, _ = global_local_centering(no_images)
    np.save(OUTPUT_NO_DATASET, no_local_centered_images)
    np.save(OUTPUT_NO_MASK, non_occluded)
    save_pixel_var(no_local_centered_images, OUTPUT_NO_VAR_FILE)

    # Third dataset: non-occluded and natural lighting (nonl)
    natural_lighting = (population['lighting'] == 'natural').as_matrix()
    nonl = natural_lighting & non_occluded
    nonl_images = local_centered_images[nonl]
    nonl_local_centered_images, _, _ = global_local_centering(nonl_images)
    np.save(OUTPUT_NONL_DATASET, nonl_local_centered_images)
    np.save(OUTPUT_NONL_MASK, nonl)
    save_pixel_var(nonl_local_centered_images, OUTPUT_NONL_VAR_FILE)
