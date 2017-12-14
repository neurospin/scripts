#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:46:01 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil


BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects"
TEMPLATE_PATH = os.path.join(BASE_PATH,"freesurfer_template")
MASK_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy"
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/sczCo/population.csv"

OUTPUT = "/neurospin/brainomics/2017_memento/analysis/FS/data/sczCo/"

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
assert n == 314
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
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (314, 327684)

mask = np.load(MASK_PATH)


X = Xtot[:, mask]
assert X.shape == (314, 299879)

#############################################################################
np.save(os.path.join(OUTPUT, "y.npy"), y)
np.save(os.path.join(OUTPUT, "X.npy"), X)

#############################################################################


