#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:10:36 2016

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil

BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_FS = '/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_assembled_data_fsaverage'
TEMPLATE_PATH = "/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_template"

INPUT_CSV = "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/population.csv"
OUTPUT = "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data"
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
assert n == 123
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop["IRM.1"]):
    cur = pop[pop["IRM.1"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["group.num"]
    Z[i, 1:] = np.asarray(cur[["Âge", "sex.num"]]).ravel()
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (123, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() ==  317102

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape == (123,317102)


#############################################################################

X = np.hstack([Z, X])
assert X.shape == (123, 317105)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (123, 317105)


# Center/scale
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
X = np.nan_to_num(X)
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i surface points' % \
    (n, p, mask.sum()))
fh.close()

np.save(os.path.join(OUTPUT, "y.npy"), y)

##############################################################################
#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299731, 299731) for a in Atv])