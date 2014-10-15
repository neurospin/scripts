# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 14:13:15 2014

@author: cp243490

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask for PET files (PET_j0Scaled_images)

INPUT:
- /neurospin/brainomics/2014_deptms/clinic/deprimPetInfo.csv

OUTPUT:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel

"""

import os
import numpy as np
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil

#import proj_classif_config
REP_MAP = {'norep': 0, 'rep': 1}

BASE_PATH = "/neurospin/brainomics/2014_deptms"

INPUT_CSV = os.path.join(BASE_PATH,          "clinic", "deprimPetInfo.csv")

OUTPUT_CSI = os.path.join(BASE_PATH,         "pet_wb")
OUTPUT_ATLAS = os.path.join(BASE_PATH,       "pet_wb_gtvenet")
OUTPUT_CS_ATLAS = os.path.join(BASE_PATH,    "pet_cs_gtvenet")

if not os.path.exists(OUTPUT_CSI):
    os.makedirs(OUTPUT_CSI)
if not os.path.exists(OUTPUT_ATLAS):
    os.makedirs(OUTPUT_ATLAS)
if not os.path.exists(OUTPUT_CS_ATLAS):
    os.makedirs(OUTPUT_CS_ATLAS)

# Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")
pop['rep_norep.num'] = pop["rep_norep"].map(REP_MAP)

#############################################################################
# Read images
n = len(pop)
assert n == 34
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i, PET_file in enumerate(pop['PET_file']):
    cur = pop[pop.PET_file == PET_file]
    print cur
    imagefile_name = os.path.join(BASE_PATH,
                                   "images",
                                   "PET_j0Scaled_images",
                                   PET_file)
    babel_image = nibabel.load(imagefile_name)
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Sex"]]).ravel()
    y[i, 0] = cur["rep_norep.num"]

shape = babel_image.get_data().shape
#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 568820

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name,
    output=os.path.join(OUTPUT_ATLAS, "mask.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 277016
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 275514
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_ATLAS, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT_ATLAS, "mask.nii"))
assert np.all(mask_atlas == im.get_data())


shutil.copyfile(os.path.join(OUTPUT_ATLAS, "mask.nii"), os.path.join(OUTPUT_CS_ATLAS, "mask.nii"))

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
assert mask_bool.sum() == 275514
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT_CSI, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))

#############################################################################
# Xcsi
X = Xtot[:, mask_bool.ravel()]
X = np.hstack([Z, X])
assert X.shape == (34, 275517)
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
np.save(os.path.join(OUTPUT_CSI, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CSI, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

# atlas
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z, X])
assert X.shape == (34, 275517)
n, p = X.shape
np.save(os.path.join(OUTPUT_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

# atlas cs
X = Xtot[:, (mask_atlas.ravel() != 0)]
X = np.hstack([Z[:, 1:], X])
assert X.shape == (34, 275517)
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT_CS_ATLAS, "X.npy"), X)
fh = open(os.path.join(OUTPUT_CS_ATLAS, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, (mask_atlas.ravel() != 0).sum()))
fh.close()

np.save(os.path.join(OUTPUT_CSI, "y.npy"), y)
np.save(os.path.join(OUTPUT_ATLAS, "y.npy"), y)
np.save(os.path.join(OUTPUT_CS_ATLAS, "y.npy"), y)

#############################################################################
# MULM
mask_ima = nibabel.load(os.path.join(OUTPUT_CSI, "mask.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(OUTPUT_CSI, "X.npy"))
y = np.load(os.path.join(OUTPUT_CSI, "y.npy"))
Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

from mulm import MUOLSStatsCoefficients
muols = MUOLSStatsCoefficients()
muols.fit(X=DesignMat, Y=Y)

tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=Y, contrast=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "t_stat_rep_min_norep_pet_wb.nii.gz"))

thres = .1
m1 = pvals <= thres
m2 = (pvals > thres) & (pvals < (1. - thres))
m3 = pvals >= (1. - thres)
print np.sum(m1), np.sum(m2), np.sum(m3)
arr = np.zeros(mask_arr.shape)
val = np.zeros(pvals.shape, dtype=int)
val[m1] = 1.
val[m2] = 2.
val[m3] = 3.
arr[mask_arr] = val
arr = brainomics.image_atlas.smooth_labels(arr, size=(3, 3, 3))
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "pval-quantile_rep_min_norep_pet_wb.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, "pval_rep_min_norep_pet_wb.nii.gz"))

# anatomist /neurospin/brainomics/2014_deptms/pet_wb/*.nii.gz'''