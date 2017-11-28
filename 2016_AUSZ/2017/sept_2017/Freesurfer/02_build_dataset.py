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


INPUT_FS = '/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_assembled_data_fsaverage'
TEMPLATE_PATH = "/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_template"

INPUT_CSV = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/population.csv"
OUTPUT = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data"
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
Z = np.zeros((n, 2)) # Age + Gender
MASCtot = np.zeros((n, 1)) # MASCtot scores

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
    Z[i, :] = np.asarray(cur[["Âge", "sex.num"]]).ravel()
    MASCtot[i, 0] = cur[" MASCtot"]
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
assert X.shape == (123, 317104)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (123, 317104)


np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT, "MASCtot.npy"), MASCtot)


np.save(os.path.join(OUTPUT, "X_patients.npy"), X[y!=0,:])
np.save(os.path.join(OUTPUT, "y_patients.npy"), y[y!=0])
np.save(os.path.join(OUTPUT, "MASCtot_patients.npy"), MASCtot[y!=0])

##############################################################################
#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (317102, 317102) for a in Atv])


###############################################################################
#0 : controls
#1: ASD
#2 SCZ-ASD
#3 SCZ

##########################
#ASD vs CONTROLS (1 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/asd_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 2,:]
y = y[y!= 2]
X = X[y!= 3,:]
y = y[y!= 3]
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz

##########################
#scz_asd_vs_asd (2 vs 1)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/scz_asd_vs_asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 0,:]
y = y[y!= 0]
X = X[y!= 3,:]
y = y[y!= 3]
y[y==2]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz-asd and 1 = asd

##########################
#scz_asd_vs_controls(2 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/scz_asd_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 3,:]
y = y[y!= 3]
y[y==2]=1
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz-asd

##########################
#scz vs asd(3 vs 1)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/scz_vs_asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 0,:]
y = y[y!= 0]
X = X[y!= 2,:]
y = y[y!= 2]
y[y==3]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz and 1 = asd

##########################
#scz vs controls(3 vs 0)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/scz_vs_controls"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 2,:]
y = y[y!= 2]
y[y==3]=1
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = controls and 1 = scz

##########################
#scz vs scz-asd(3 vs 2)
##########################
#########################
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer/data/scz_vs_scz-asd"
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
X = X[y!= 1,:]
y = y[y!= 1]
X = X[y!= 0,:]
y = y[y!= 0]
y[y==3]=1
y[y==2]=0
np.save(os.path.join(WD,'X.npy'),X)
np.save(os.path.join(WD,'y.npy'),y)
#: 0 = scz-asd and 1 = scz