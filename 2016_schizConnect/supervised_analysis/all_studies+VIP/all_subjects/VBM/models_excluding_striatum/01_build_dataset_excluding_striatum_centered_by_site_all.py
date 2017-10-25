#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:17:34 2017

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
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
import mulm
import sklearn
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
ATLAS_PATH = "/neurospin/brainomics/2016_schizConnect/atlas"
INPUT_ROIS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_centered_excluding_striatum/ROI_labels.csv"
OUTPUT = os.path.join(BASE_PATH,"data","data_centered_excluding_striatum")
penalty_start = 2



#Striatum = putamen + Caudate + Pallidum
#5 6 16 17
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 606
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i,:] = np.asarray(cur[["age", "sex_num"]]).ravel()
    y[i, 0] = cur["dx_num"]

shape = babel_image.get_data().shape


sub_image = nibabel.load(os.path.join(ATLAS_PATH,"HarvardOxford-sub-maxprob-thr0-1.5mm.nii.gz"))
sub_arr = sub_image.get_data()

#Labels of striatum : https://neurovault.org/images/1700/
caudate = np.logical_or(sub_arr == 5,sub_arr == 16)
putamen = np.logical_or(sub_arr == 6,sub_arr == 17)
pallidum = np.logical_or(sub_arr == 7,sub_arr == 18)
striatum = np.logical_or(caudate,putamen)
striatum = np.logical_or(striatum,pallidum)

out_im = nibabel.Nifti1Image(striatum.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_striatum.nii"))
#############################################################################
# Compute mask
# Implicit Masking
Xtot = np.vstack(images)

mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 150811

#Remove Striatum voxels from mask
mask_without_striatum = np.logical_and(mask,np.logical_not(striatum))
assert mask_without_striatum.sum() == 146142
#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT, "mask_without_striatum.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask_without_striatum)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 120852
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 120852
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]


site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")

X[site==1,:] = X[site==1,:] - X[site==1,:].mean(axis=0)
X[site==2,:] = X[site==2,:] - X[site==2,:].mean(axis=0)
X[site==3,:] = X[site==3,:] - X[site==3,:].mean(axis=0)
X[site==4,:] = X[site==4,:] - X[site==4,:].mean(axis=0)


X = np.hstack([Z, X])
assert X.shape == (606, 120854)

#Remove nan lines
X = X[np.logical_not(np.isnan(y)).ravel(),:]
y = y[np.logical_not(np.isnan(y))]
assert X.shape == (606, 120854)


np.save(os.path.join(OUTPUT, "X.npy"),X)
np.save(os.path.join(OUTPUT, "y.npy"),y)


###############################################################################
# precompute linearoperator

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 11.909770107366217)

###############################################################################
