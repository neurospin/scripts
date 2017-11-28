# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil

BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS"
INPUT_FS = "/neurospin/brainomics/2017_memento/analysis/FS/freesurfer_assembled_data_fsaverage"
TEMPLATE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/freesurfer_template"
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"

OUTPUT = os.path.join(BASE_PATH,"data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
assert  pop.shape == (2164, 27)

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
assert n == 2164
surfaces = list()
for i, ID in enumerate(pop['usubjid']):
    cur = pop[pop["usubjid"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    print (i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (2164, 327684)

#USE A MASK to restrict analysis to voxel of cortical thickness
FS_mask_path = "/neurospin/brainomics/neuroimaging_ressources/freesurfer_utils/fsaverage"
mask_surf_r_path = os.path.join(FS_mask_path, "rh.mask.surf.gii")
mask_surf_l_path = os.path.join(FS_mask_path, "lh.mask.surf.gii")

# FS cortical mask
from nibabel import gifti
mask_surf_l = gifti.read(mask_surf_l_path).darrays[0].data.astype(bool)
mask_surf_r = gifti.read(mask_surf_r_path).darrays[0].data.astype(bool)
mask_surf = np.hstack([mask_surf_l, mask_surf_r])


X = Xtot[:, mask_surf]
assert X.shape == (2164, 299879)

mask_implicit = ((np.max(X, axis=0) > 0) & (np.std(X, axis=0) > 1e-2))


X = X[:,mask_implicit ]
assert X.shape == (2164, 299879)

np.save(os.path.join(OUTPUT, "mask.npy"), mask_surf)
np.save(os.path.join(OUTPUT, "X.npy"), X)

#############################################################################
import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask_surf, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)
assert np.all([a.shape == (299879, 299879) for a in Atv])