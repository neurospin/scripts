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


BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei/september_2017/VBM/ICAAR_sczco'
INPUT_CSV= os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")


# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#############################################################################
# Read images
n = len(pop)
assert n == 55
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, :] = np.asarray(cur[["age","sex.num"]]).ravel()
    y[i, 0] = cur["group_outcom.num"]

shape = babel_image.get_data().shape


Xtot = np.vstack(images)
assert Xtot.shape == (55, 2122945)

Xtot= Xtot[np.logical_not(np.isnan(y)).ravel(),:]
Z= Z[np.logical_not(np.isnan(y)).ravel()]

y=y[np.logical_not(np.isnan(y))]
assert Xtot.shape == (41, 2122945)

#center by site
Xtot = Xtot - Xtot.mean(axis=0)

#Load scjizconnect mask
mask_SczCo = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/mask.nii"
mask_nib = nibabel.load(mask_SczCo)

shutil.copy(mask_SczCo, OUTPUT)
mask = mask_nib.get_data().ravel()
assert mask.sum() == 125959

X = Xtot[:, (mask==1)]
assert X.shape == (41, 125959)

#############################################################################
X = np.hstack([Z, X])
assert X.shape == (41, 125961)
#Remove nan lines

np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)

###############################################################################
###############################################################################
## precompute linearoperator
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))

mask = nibabel.load(os.path.join(OUTPUT, "mask.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
