# -*- coding: utf-8 -*-
"""
@author: edouard.duchesnay@cea.fr

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT:
- mask.nii.gz
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs"


INPUT_CSV = os.path.join(BASE_PATH,          "ADRS", "population.csv")

OUTPUT = os.path.join(BASE_PATH,             "ADRS")
OUTPUT_CS = os.path.join(BASE_PATH,          "ADRS_cs")
#OUTPUT_ATLAS = os.path.join(BASE_PATH,       "ADRS_gtvenet")
#OUTPUT_CS_ATLAS = os.path.join(BASE_PATH,    "ADRS_cs_gtvenet")

if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
if not os.path.exists(OUTPUT_CS): os.makedirs(OUTPUT_CS)
#os.makedirs(OUTPUT_ATLAS)
#os.makedirs(OUTPUT_CS_ATLAS)

# Read input subjects

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['Gender_num'] = pop["Gender"].map(GENDER_MAP)

#############################################################################
# Read images
n = len(pop)
assert n == 1082
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i in xrange(n):
    cur = pop.iloc[i]
    print cur
    babel_image = nibabel.load(cur["mri_path"])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Gender_num"]]).ravel()
    y[i, 0] = cur["adrs"]

shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
np.save(os.path.join(OUTPUT, "Xtot.npy"), Xtot)
os.exit(0)
Xtot = np.load(os.path.join(OUTPUT, "Xtot.npy"))

del images
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 313734

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=cur["mri_path"],
    output=os.path.join("/tmp", "mask.nii.gz"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 638715
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 285983
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join("/tmp", "mask.nii.gz"))
im = nibabel.load(os.path.join("/tmp", "mask.nii.gz"))
assert np.all(mask_atlas == im.get_data())


#shutil.copyfile(os.path.join(OUTPUT_ATLAS, "mask.nii.gz"), os.path.join(OUTPUT_CS_ATLAS, "mask.nii.gz"))

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
assert mask_bool.sum() == 285983
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii.gz"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii.gz"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))

shutil.copyfile(os.path.join(OUTPUT, "mask.nii.gz"), os.path.join(OUTPUT_CS, "mask.nii.gz"))

#############################################################################
# X
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z, X])
assert X.shape == (242, 285986)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_bool.sum()))
fh.close()

# Xcs
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (242, 285985)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

## atlas
#X = Xtot[:, (mask_atlas.ravel() != 0)]
#X = np.hstack([Z, X])
#assert X.shape == (242, 285986)
#n, p = X.shape
#np.save(os.path.join(OUTPUT_ATLAS, "X.npy"), X)
#fh = open(os.path.join(OUTPUT_ATLAS, "X.npy").replace("npy", "txt"), "w")
#fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
#    (n, p, (mask_atlas.ravel() != 0).sum()))
#fh.close()
#
## atlas cs
#X = Xtot[:, (mask_atlas.ravel() != 0)]
#X = np.hstack([Z[:, 1:], X])
#assert X.shape == (242, 285985)
#X -= X.mean(axis=0)
#X /= X.std(axis=0)
#n, p = X.shape
#np.save(os.path.join(OUTPUT_CS_ATLAS, "X.npy"), X)
#fh = open(os.path.join(OUTPUT_CS_ATLAS, "X.npy").replace("npy", "txt"), "w")
#fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
#    (n, p, (mask_atlas.ravel() != 0).sum()))
#fh.close()

np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS, "y.npy"), y)
#np.save(os.path.join(OUTPUT_ATLAS, "y.npy"), y)
#np.save(os.path.join(OUTPUT_CS_ATLAS, "y.npy"), y)

