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


BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")


# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 239
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
site = np.zeros((n, 1))
age = np.zeros((n, 1))
ados_tot = np.zeros((n, 1))
ados_sa = np.zeros((n, 1))
ados_rrb = np.zeros((n, 1))
srs_t = np.zeros((n, 1))
srs_self_t = np.zeros((n, 1))

images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, :] = np.asarray(cur[["age","sex_num"]]).ravel()
    y[i, 0] = cur["dx_num"]
    site[i, 0] = cur["site"]
    age[i, 0] = cur["age"]
    ados_tot [i, 0] = cur["ados_2_CSS_total"]
    ados_sa[i, 0] = cur["ados_2_SA_CSS"]
    ados_rrb[i, 0] = cur["ados_2_SA_CSS"]
    srs_t[i, 0] = cur["SRS_tscore"]
    srs_self_t[i, 0] = cur["SRS_tscore_self"]

shape = babel_image.get_data().shape

np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/site.npy",site.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/age.npy",age.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/ados_tot.npy",ados_tot.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/ados_sa.npy",ados_sa.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/ados_rrb.npy",ados_rrb.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/srs_t.npy",srs_t.ravel())
np.save("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/Adult_Template/data/srs_self_t.npy",srs_self_t.ravel())

#############################################################################
# Compute mask
# Implicit Masking .
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 302862

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT, "mask.nii"))

mask_atlas = babel_mask_atlas.get_data()
#assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
#assert np.sum(mask_atlas != 0) ==   249797
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
im = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 302862
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]




X[site.ravel()==1,:] = X[site.ravel()==1,:] - X[site.ravel()==1,:].mean(axis=0)
X[site.ravel()==2,:] = X[site.ravel()==2,:] - X[site.ravel()==2,:].mean(axis=0)
X[site.ravel()==3,:] = X[site.ravel()==3,:] - X[site.ravel()==3,:].mean(axis=0)
X[site.ravel()==4,:] = X[site.ravel()==4,:] - X[site.ravel()==4,:].mean(axis=0)
X[site.ravel()==5,:] = X[site.ravel()==5,:] - X[site.ravel()==5,:].mean(axis=0)
X[site.ravel()==6,:] = X[site.ravel()==6,:] - X[site.ravel()==6,:].mean(axis=0)

#Stack covariates
X = np.hstack([Z, X])


n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)

###############################################################################
###############################################################################
# precompute linearoperatorSS
#X = np.load("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data/X.npy")
#y = np.load("/neurospin/brainomics/2018_euaims_leap_predict_vbm/results/VBM/1.5mm/data/y.npy")

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
