# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 18:20:17 2014

@author: md238665

Create population file for the cropped images DB.

There is no duplicates in this base. Note that the sex is upper-case here.

We don't dump images in a numpy array because it's huge and application scripts
may subsample images.

"""

import os
import glob
import re
import collections

import pandas as pd

################
# Input/Output #
################

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/AR_faces"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,
                         "raw_data",
                         "cropped_faces")
INPUT_IMAGES_GLOB = os.path.join(INPUT_DIR,
                                 "*.bmp")
INPUT_IMAGE_RE = re.compile('(?P<sex>[WM])-(?P<id>\d{3})-(?P<expr>\d{1,2})')
INPUT_IMAGE_SHAPE = (120, 165, 3)

# Output is put in INPUT_DIR because it's the better place.
OUTPUT_DIR = os.path.join(INPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_POP = os.path.join(OUTPUT_DIR,
                          "population.csv")

##########
# Script #
##########

# Find files (sort is not necessary here)
all_images = glob.glob(INPUT_IMAGES_GLOB)
all_images.sort()
print "Found", len(all_images), "files"

# Create dataframe
file_infos = collections.OrderedDict.fromkeys(['sex', 'id', 'expr'])
for k in file_infos.keys():
    file_infos[k] = []
print "Indexing into df"
for filename in all_images:
    # Extract info from name
    name = os.path.basename(filename)
    file_info = re.match(INPUT_IMAGE_RE, name).groupdict()
    file_infos['sex'].append(file_info["sex"])
    file_infos['id'].append(int(file_info["id"]))
    file_infos['expr'].append(int(file_info["expr"]))
file_infos['file'] = all_images

images_df = pd.DataFrame.from_dict(file_infos)

# Map columns
# We could use more elaborate mappings (using modulo of expr) but it's not woth

# Session
session_map = lambda expr: 0 if expr < 14 else 1
images_df['session'] = images_df['expr'].map(session_map)

# Expression
# Possible values: NA, 'neutral', smile', 'anger', 'scream'
# We only label the explicitely labelled images (most images could be
# 'neutral').
expression_map = {1: 'neutral',
                  2: 'smile',
                  3: 'anger',
                  4: 'scream',
                  14: 'neutral',
                  15: 'smile',
                  16: 'anger',
                  17: 'scream'}
images_df['expression'] = images_df['expr'].map(expression_map,
                                                na_action='ignore')

# Occlusion
# Possible values: NA, 'sun glasses', 'scarf'
occlusion_map = {8: 'sun glasses',
                 9: 'sun glasses',
                 10: 'sun glasses',
                 11: 'scarf',
                 12: 'scarf',
                 13: 'scarf',
                 21: 'sun glasses',
                 22: 'sun glasses',
                 23: 'sun glasses',
                 24: 'scarf',
                 25: 'scarf',
                 26: 'scarf'}
images_df['occlusion'] = images_df['expr'].map(occlusion_map,
                                               na_action='ignore')

# Lightning
# Possible values: 'natural', 'left side', 'right side', 'both side'
# I decided to label cases 8 and 11 as 'natural'
def lightning_map(expr):
    expr_mod = expr - 13 if expr > 13 else expr
    if (expr_mod in (5, 9, 12)):
        return 'left side'
    if (expr_mod in (6, 10, 13)):
        return 'right side'
    if (expr_mod == 7):
        return 'both side'
    return 'natural'
images_df['lighting'] = images_df['expr'].map(lightning_map,
                                              na_action='ignore')

images_df.to_csv(OUTPUT_POP,
                 index=False)
