#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:59:22 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST"
INPUT_FS = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/freesurfer_assembled_data_fsaverage"
TEMPLATE_PATH = os.path.join(BASE_PATH,"Freesurfer","freesurfer_template")

INPUT_CSV = os.path.join(BASE_PATH,"Freesurfer","population.csv")
OUTPUT = os.path.join(BASE_PATH,"Freesurfer","data")



# Read pop csv
pop = pd.read_csv(INPUT_CSV)

#############################################################################
## Build mesh template
import brainomics.mesh_processing as mesh_utils
cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "rh.pial.gii"))
cor = np.vstack([cor_l, cor_r])
tri_r += cor_l.shape[0]
tri = np.vstack([tri_l, tri_r])
mesh_utils.mesh_from_arrays(cor, tri, path=os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))
shutil.copyfile(os.path.join(TEMPLATE_PATH, "lrh.pial.gii"), os.path.join(OUTPUT , "lrh.pial.gii"))

#############################################################################
# Read images
n = len(pop)
assert n == 260
Z = np.zeros((n, 2)) # Age + Gender #only one site, no need for a site covariate

y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop["id"]):
    cur = pop[pop["id"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["dx_num"]
    Z[i, :] = np.asarray(cur[["age", "sex_num"]]).ravel()
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (260, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() ==  299794

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape ==  (260, 299794)


#############################################################################

X = np.hstack([Z, X])
assert X.shape == (260, 299796)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (260, 299796)



np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)


#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299794, 299794) for a in Atv])
