# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:46:15 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil


#import proj_classif_config
GENDER_MAP = {'female': 0, 'male': 1}

BASE_PATH = "/neurospin/brainomics/2016_pca_struct/adni"

INPUT_FS = os.path.join(BASE_PATH,"freesurfer_assembled_data_fsaverage")

TEMPLATE_PATH = os.path.join(BASE_PATH, "data/freesurfer_template")

INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
OUTPUT = os.path.join(BASE_PATH,"data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['PTGENDER.num'] = pop["Sex"].map(GENDER_MAP)

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
assert n == 360
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop['Subject ID']):
    cur = pop[pop["Subject ID"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["DX.num"]
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (360, 327684)
#assert Xtot.shape == (360, 81924)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
#assert mask.sum() == 317379

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

# Xcs
X = Xtot[:, mask]

#############################################################################
# Some basic stat before centering/scaling
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)


#############################################################################
# Center/scale
X -= X.mean(axis=0)
X /= X.std(axis=0)
n, p = X.shape
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('Centered and scaled data. Shape = (%i, %i): Age + Gender + %i surface points' % \
    (n, p, mask.sum()))
fh.close()

np.save(os.path.join(OUTPUT, "y.npy"), y)

#############################################################################
#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.utils.linalgs
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
#assert np.all([a.shape == (299731, 299731) for a in Atv])
