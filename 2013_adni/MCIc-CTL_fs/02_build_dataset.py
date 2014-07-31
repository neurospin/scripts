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

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
INPUT_IMAGE_DIR = os.path.join(BASE_PATH, "freesurfer_assembled_data")
TEMPLATE_PATH = os.path.join(BASE_PATH, "freesurfer_template")


INPUT_CSV = os.path.join(BASE_PATH,          "MCIc-CTL_fs", "population.csv")

OUTPUT = os.path.join(BASE_PATH,             "MCIc-CTL_fs")

# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['PTGENDER.num'] = pop["PTGENDER"].map(GENDER_MAP)

#############################################################################
## Build mesh template
import brainomics.mesh_processing as mesh_utils
cor_l, tri_l = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "lh.pial.gii"))
cor_r, tri_r = mesh_utils.mesh_arrays(os.path.join(TEMPLATE_PATH, "rh.pial.gii"))
cor = np.vstack([cor_l, cor_r])
tri_r += cor_l.shape[0]
tri = np.vstack([tri_l, tri_r])
mesh_utils.mesh_from_arrays(cor, tri, path=os.path.join(TEMPLATE_PATH, "lrh.pial.gii"))

#############################################################################
# Read images
n = len(pop)
assert n == 201
Z = np.zeros((n, 2)) # Age + Gender
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, PTID in enumerate(pop['PTID']):
    cur = pop[pop.PTID == PTID]
    print cur
    #cur = pop.iloc[0]
    left = nb.load(cur["mri_path_lh"].values[0]).get_data().ravel()
    right = nb.load(cur["mri_path_rh"].values[0]).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    Z[i, :] = np.asarray(cur[["AGE", "PTGENDER.num"]]).ravel()
    y[i, 0] = cur["DX.num"]


Xtot = np.vstack(surfaces)
assert Xtot.shape == (201, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() == 317089

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

# Xcs
X = Xtot[:, mask]
X = np.hstack([Z, X])
assert X.shape == (201, 317091)

#############################################################################
# Some basic stat before centering/scaling
means = np.mean(X[:, 2:], axis=0)
stds = np.std(X[:, 2:], axis=0)
mins = np.min(X[:, 2:], axis=0)
maxs = np.max(X[:, 2:], axis=0)
print "Means:", means.min(), means.max(), means.mean()
print "Std:",  stds.min(), stds.max(), stds.mean()
print "Mins:", mins.min(), mins.max(), mins.mean()
print "Maxs:", maxs.min(), maxs.max(), maxs.mean(), (maxs == 0).sum()
#In [93]: Means: 0.00120785424662 4.70434341146 2.20050826134
#In [94]: Std: 0.0170816385694 2.18101036803 0.593103047434
#In [95]: Mins: 0.0 3.02653193474 0.936884414047
#In [96]: Maxs: 0.24277870357 5.0 3.94083362002 0


arr = np.zeros(mask.shape); arr[mask] = means
mesh_utils.save_texture(path=os.path.join(OUTPUT, "mean.gii"), data=arr, intent='NIFTI_INTENT_NONE')
arr = np.zeros(mask.shape); arr[mask] = stds
mesh_utils.save_texture(path=os.path.join(OUTPUT, "std.gii"), data=arr, intent='NIFTI_INTENT_NONE')
arr = np.zeros(mask.shape); arr[mask] = maxs
mesh_utils.save_texture(path=os.path.join(OUTPUT, "max.gii"), data=arr, intent='NIFTI_INTENT_NONE')

# anatomist mean.gii std.gii max.gii ../freesurfer_template/lrh.pial.gii
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

#############################################################################
# MULM

DesignMat = np.zeros((Z.shape[0], 4)) # y, age, sex, intercept
DesignMat[:, 0] = y.ravel()
DesignMat[:, 1:3] = Z
DesignMat[:, 3] = 1.

from mulm import MUOLSStatsCoefficients
muols = MUOLSStatsCoefficients()
muols.fit(X=DesignMat, Y=X[:, 2:])
tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=X[:, 2:], contrast=[-1, 0, 0, 0], pval=True)

arr = np.zeros(mask.shape); arr[mask] = tvals
mesh_utils.save_texture(path=os.path.join(OUTPUT, "t_stat_CTL-MCIc.gii"), data=arr, intent='NIFTI_INTENT_TTEST')

arr = np.zeros(mask.shape); arr[mask] = pvals
mesh_utils.save_texture(path=os.path.join(OUTPUT, "pval_CTL-MCIc.gii"), data=arr, intent='NIFTI_INTENT_PVAL')

tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=X[:, 2:], contrast=[1, 0, 0, 0], pval=True)

arr = np.zeros(mask.shape); arr[mask] = tvals
mesh_utils.save_texture(path=os.path.join(OUTPUT, "t_stat_MCIc-CTL.gii"), data=arr, intent='NIFTI_INTENT_TTEST')

arr = np.zeros(mask.shape); arr[mask] = pvals
mesh_utils.save_texture(path=os.path.join(OUTPUT, "pval_MCIc-CTL.gii"), data=arr, intent='NIFTI_INTENT_PVAL')

# anatomist mean.gii std.gii max.gii t_stat.gii ../freesurfer_template/lrh.pial.gii
