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

INPUT_CSV = os.path.join(BASE_PATH,"Freesurfer","population_50yo.csv")
OUTPUT = os.path.join(BASE_PATH,"Freesurfer","data","50yo")
penalty_start = 2

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
assert n == 226
Z = np.zeros((n, penalty_start)) # Age + Gender #only one site, no need for a site covariate
# Z = np.zeros((n, 3)) # Intercept + Age + Gender #only one site, no need for a site covariate
# Z[:, 0] = 1 # Intercept
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
    #Z[i, 1:] = np.asarray(cur[["age", "sex_num"]]).ravel()
    Z[i, :] = np.asarray(cur[["age", "sex_num"]]).ravel()
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (226, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() == 299794

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape ==  (226, 299794)


#############################################################################

X = np.hstack([Z, X])
assert X.shape == (226, 299796)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y = y[np.logical_not(np.isnan(y))]
assert X.shape == (226, 299796)

X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape

np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)

###############################################################################
# precompute linearoperator

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
cor, tri = mesh_utils.mesh_arrays(os.path.join(OUTPUT , "lrh.pial.gii"), nibabel=True)

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.9994390558194546, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299796-penalty_start, 299796-penalty_start) for a in Atv])

###############################################################################
# precompute beta start
import parsimony.estimators as estimators
from sklearn import preprocessing
import time

X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))

betas = dict()

alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]

for alpha in alphas:
    mod = estimators.RidgeLogisticRegression(l=alpha, class_weight="auto",
                                             penalty_start=penalty_start,
                                             algorithm_params=dict(max_iter=10000))
    t_ = time.clock()
    mod.fit(X, y.ravel())
    print(time.clock() - t_, mod.algorithm.num_iter) # 11564
    betas["lambda_%.4f" % alpha] = mod.beta

np.savez(os.path.join(OUTPUT, "beta_start.npz"), **betas)


beta_start = np.load(os.path.join(OUTPUT, "beta_start.npz"))
assert np.all([np.all(beta_start[a] == betas[a]) for a in beta_start.keys()])


