# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:03:19 2014

@author: md238665

Create population file for the static sequences DB.

Convert images to PNG and create CSV.
The RAW format stores the color plane first. Most image vizualization tools
can't display that and to load them in numpy/scipy we should have perform some
manipulations). Therefore it's easier to convert the images to PNG before.

One of the problem is that some conditions are duplicated so we try to ignore
them. Normally the chosen image should be the first in the lexicographic sort

We don't dump images a numpy array because it's huge and application scripts
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
                         "static_sequences")
INPUT_IMAGES_GLOB = os.path.join(INPUT_DIR,
                                 "*",
                                 "*.raw")
CONVERT_COMMAND = """convert -depth 8 -interlace plane -size 768x576 """ \
                   """rgb:{input_file} {output_file}"""
INPUT_IMAGE_RE = re.compile('(?P<sex>[wm])-(?P<id>\d{3})-(?P<expr>\d{1,2})')
INPUT_IMAGE_SHAPE = (576, 768, 3)

# Output is put in INPUT_DIR because it's the better place.
OUTPUT_DIR = os.path.join(INPUT_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_DUPES = os.path.join(OUTPUT_DIR,
                            "dupes.txt")
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
        print >>f, "{k}:".format(k=k), " ".join([str(e) for e in v])

# Create dataframe (we could do that at the same time than conversion but
# it's rather fast).
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

# Remove duplicates (as the image filenames are sorted, the chosen image for
# duplicates should be the first in lexicographic order).
images_df = images_df[~images_df.duplicated(subset=["sex", "id", "expr"])]
images_df.sort(columns=["sex", "id", "expr"],
               inplace=True)
assert(len(images_df) == n_uniques)
images_df.to_csv(OUTPUT_POP,
                 index=False)
