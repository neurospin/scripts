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

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects"
TEMPLATE_PATH = os.path.join(BASE_PATH,"freesurfer_template")
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data","mean_centered_by_site_all")


# Read pop csv
pop = pd.read_csv(INPUT_CSV)
#np.save('/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy',pop["site_num"].as_matrix())

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
assert n == 567
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
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
assert Xtot.shape == (567, 327684)

site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy")

Xtot[site==1,:] = Xtot[site==1,:] - Xtot[site==1,:].mean(axis=0)
Xtot[site==2,:] = Xtot[site==2,:] - Xtot[site==2,:].mean(axis=0)
Xtot[site==3,:] = Xtot[site==3,:] - Xtot[site==3,:].mean(axis=0)
Xtot[site==4,:] = Xtot[site==4,:] - Xtot[site==4,:].mean(axis=0)



mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() ==  299862

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape == (567, 299862)

#############################################################################
X = np.hstack([Z, X])
assert X.shape == (567, 299864)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (567, 299864)

np.save(os.path.join(OUTPUT, "X.npy"), X)
np.save(os.path.join(OUTPUT, "y.npy"), y)


#############################################################################


site = X[:,2]
X = np.hstack([Z[:,:2], X[:,3:]])
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_by_site"
#NUDAST
X3 = X[site==3,:]
y3 = y[site==3]
np.save(os.path.join(WD,"NUSDAST", "X.npy"), X3)
np.save(os.path.join(WD,"NUSDAST", "y.npy"), y3)

#NCOBRE
X1 = X[site==1,:]
y1 = y[site==1]
np.save(os.path.join(WD,"COBRE", "X.npy"), X1)
np.save(os.path.join(WD,"COBRE", "y.npy"), y1)


#NMORPHCH
X2 = X[site==2,:]
y2 = y[site==2]
np.save(os.path.join(WD,"NMORPH", "X.npy"), X2)
np.save(os.path.join(WD,"NMORPH", "y.npy"), y2)

#VIP
X4 = X[site==4,:]
y4 = y[site==4]
np.save(os.path.join(WD,"VIP", "X.npy"), X4)
np.save(os.path.join(WD,"VIP", "y.npy"), y4)











#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299862, 299862) for a in Atv])
