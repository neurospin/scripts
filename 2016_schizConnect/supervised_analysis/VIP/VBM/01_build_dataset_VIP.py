#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:20:32 2017


5@author: ad24740

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- population.csv

OUTPUT:
- mask.nii
- y.npy
- X.npy = Age + Gender + Voxels
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil
import mulm
import sklearn


BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")


# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 92
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, :] = np.asarray(cur[["sex_code"]]).ravel()
    y[i, 0] = cur["dx"]

shape = babel_image.get_data().shape



#############################################################################
# Compute mask
# Implicit Masking .
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 269964

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT, "mask.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) ==   249797
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() ==  249797
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]

#Stack covariates
X = np.hstack([Z, X])
assert X.shape == (92,  249799)


n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)

###############################################################################
###############################################################################
## precompute linearoperator
#X = np.load(os.path.join(OUTPUT, "X.npy"))
#y = np.load(os.path.join(OUTPUT, "y.npy"))
#
#mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
#
#import parsimony.functions.nesterov.tv as nesterov_tv
#from parsimony.utils.linalgs import LinearOperatorNesterov
#
#Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
#Atv.save(os.path.join(OUTPUT, "Atv.npz"))
#Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
#assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
