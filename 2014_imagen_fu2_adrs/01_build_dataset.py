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
# Compute implicit mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
np.save(os.path.join(OUTPUT, "Xtot.npy"), Xtot)
del images

#os.exit(0)
Xtot = np.load(os.path.join(OUTPUT, "Xtot.npy"))

mask_implicit = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask_implicit = mask.reshape(shape)
assert mask_implicit.sum() == 730646
mask_implicit = mask_implicit.reshape(shape)
out_im = nibabel.Nifti1Image(mask_implicit.astype(int),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_implicit.nii.gz"))
out_im.to_filename(os.path.join(OUTPUT_CS, "mask_implicit.nii.gz"))


#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=cur["mri_path"],
    output=os.path.join(OUTPUT, "mask_atlas.nii.gz"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 638715
mask_atlas[np.logical_not(mask_implicit)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 625897
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_atlas.nii.gz"))
im = nibabel.load(os.path.join(OUTPUT, "mask_atlas.nii.gz"))
assert np.all(mask_atlas.astype(int) == im.get_data())
shutil.copyfile(os.path.join(OUTPUT, "mask_atlas.nii.gz"), os.path.join(OUTPUT_CS, "mask_atlas.nii.gz"))


#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_atlas_binarized = mask_atlas != 0
assert mask_atlas_binarized.sum() == 625897
out_im = nibabel.Nifti1Image(mask_atlas_binarized.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_atlas_binarized.nii.gz"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask_atlas_binarized.nii.gz"))
assert np.all(mask_atlas_binarized == (babel_mask.get_data() != 0))
shutil.copyfile(os.path.join(OUTPUT, "mask_atlas_binarized.nii.gz"), os.path.join(OUTPUT_CS, "mask_atlas_binarized.nii.gz"))

#############################################################################
# X
X = Xtot[:, mask_atlas_binarized.ravel()]
#############################################################################
# BASIC MULM
from mulm import MUOLSStatsCoefficients
muols = MUOLSStatsCoefficients()
X_design = np.hstack([y, Z])
muols.fit(X_design, X)
tvals, pvals = muols.stats(X_design, X)
p_log10 = - np.log10(pvals)
arr = np.zeros(shape)
arr[mask_atlas_binarized] = p_log10[0, :]
out_im = nibabel.Nifti1Image(arr, affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "pval_-log10_adrs.nii.gz"))

# ROI
# 70% Frontal Medial Cortex 25
#np.max(arr) 6.159

X = np.hstack([Z, X])
assert X.shape == (242, 285986)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_atlas_binarized.sum()))
fh.close()

# Xcs
X = Xtot[:, mask_atlas_binarized.ravel()]
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

