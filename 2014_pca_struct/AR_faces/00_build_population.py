# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:03:19 2014

@author: md238665

Convert images to PNG and dump a grayscale version in a numpy array.
The RAW format stores the color plane first. Most image vizualization tools
can't display that and to load them in numpy/scipy we should have perform some
manipulations). Therefore it's easier to convert the images to PNG before.

One of the problem is that some conditions are duplicated so we try to ignore
them.

"""

import os
import glob
import re
import collections

import numpy as np
import scipy.misc
import skimage
import skimage.color
import pandas as pd

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/AR_faces"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "raw_data",
                         "uncompressed")
INPUT_IMAGES_GLOB = os.path.join(INPUT_DIR,
                                 "*",
                                 "*.raw")
CONVERT_COMMAND = """convert -depth 8 -interlace plane -size 768x576 """ \
                   """rgb:{input_file} {output_file}"""
INPUT_IMAGE_RE = re.compile('(?P<sex>[wm])-(?P<id>\d{3})-(?P<expr>\d{1,2})')
INPUT_IMAGE_SHAPE = (576, 768, 3)

OUTPUT_BASE_DIR = INPUT_BASE_DIR
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR,
                          "dataset")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DUPES = os.path.join(OUTPUT_DIR,
                            "dupes.txt")
OUTPUT_X = os.path.join(OUTPUT_DIR,
                        "X.npy")
OUTPUT_POP = os.path.join(OUTPUT_DIR,
                          "population.csv")

##########
# Script #
##########

# Find files: we use sort to match more closely the filesystem
all_raw_images = glob.glob(INPUT_IMAGES_GLOB)
all_raw_images.sort()
print "Found", len(all_raw_images), "files"

# Convert images
print "Converting to PNG"
all_png_images = []
for image in all_raw_images:
    dirname, basename = os.path.split(image)
    new_basename = basename.replace('.raw', '.png')
    output_image = os.path.join(dirname, new_basename)
    cmd = CONVERT_COMMAND.format(input_file=image,
                                 output_file=output_image)
    if not os.path.exists(output_image):
        os.system(cmd)
    all_png_images.append(output_image)

# Find duplicates (for some reasons it's very hard to determine here a list
# of uniques values)
images_comp = [os.path.split(im) for im in all_png_images]
dupes = {}
for t1 in images_comp:
    for t2 in images_comp:
        if (t1[1] == t2[1]) and (t1[0] != t2[0]):
            f1 = os.path.join(*t1)
            f2 = os.path.join(*t2)
            print f1, "and", f2, "are dupes"
            k = t1[1]
            if k not in dupes:
                dupes[k] = [f1, f2]
            else:
                if f1 not in dupes[k]:
                    dupes[k].append(f1)
                if f2 not in dupes[k]:
                    dupes[k].append(f2)
n_dupes = sum([len(l) for l in dupes.values()])
n_uniques = len(all_raw_images) - n_dupes + len(dupes)
# Store dict in file
with open(OUTPUT_DUPES, "w") as f:
    for k, v in dupes.items():
        print >>f, k, " ".join([str(e) for e in v])

# Create dataframe
file_infos = collections.OrderedDict.fromkeys(['sex', 'id', 'expr'])
for k in file_infos.keys():
    file_infos[k] = []
print "Indexing into df"
for filename in all_png_images:
    # Extract info from name
    name = os.path.basename(filename)
    file_info = re.match(INPUT_IMAGE_RE, name).groupdict()
    file_infos['sex'].append(file_info["sex"])
    file_infos['id'].append(int(file_info["id"]))
    file_infos['expr'].append(int(file_info["expr"]))
file_infos['file'] = all_png_images

images_df = pd.DataFrame.from_dict(file_infos)
images_df = images_df[~images_df.duplicated(subset=["sex", "id", "expr"])]
images_df.sort(columns=["sex", "id", "expr"],
               inplace=True)
assert(len(images_df) == n_uniques)
images_df.to_csv(OUTPUT_POP)

# Put images in array
print "Convert to grayscale and store in numpy array"
images = np.empty((n_uniques, INPUT_IMAGE_SHAPE[0], INPUT_IMAGE_SHAPE[1]))
for i, image in enumerate(images_df['file']):
    # Read image
    rgb_data = scipy.misc.imread(image)
    float_data = skimage.color.rgb2gray(rgb_data)
    images[i] = float_data
np.save(OUTPUT_X, images)
