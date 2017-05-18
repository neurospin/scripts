# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:06:18 2016

@author: ad247405
"""


import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import mulm
from mulm import MUOLS
import brainomics
from brainomics import create_texture

BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei'
INPUT_FS = '/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_assembled_data_fsaverage'
TEMPLATE_PATH = os.path.join(BASE_PATH, "preproc_FS/freesurfer_template")

INPUT_CSV = os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","population.csv")
OUTPUT = os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","data")

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
assert n == 52
Z = np.zeros((n, 2)) #Age,sex
y = np.zeros((n, 1)) # DX
surfaces = list()

for i, ID in enumerate(pop['image']):
    cur = pop[pop["image"] == ID]
    mri_path_lh = cur["mri_path_lh"].values[0]
    mri_path_rh = cur["mri_path_rh"].values[0]
    left = nb.load(mri_path_lh).get_data().ravel()
    right = nb.load(mri_path_rh).get_data().ravel()
    surf = np.hstack([left, right])
    surfaces.append(surf)
    y[i, 0] = cur["group_outcom.num"]
    Z[i, :] = np.asarray(cur[["age", "sex.num"]]).ravel()
    print(i)


Xtot = np.vstack(surfaces)
assert Xtot.shape == (52, 327684)

mask = ((np.max(Xtot, axis=0) > 0) & (np.std(Xtot, axis=0) > 1e-2))
assert mask.sum() == 316308

np.save(os.path.join(OUTPUT, "mask.npy"), mask)

X = Xtot[:, mask]
assert X.shape == (52, 316308)


#############################################################################

X = np.hstack([Z, X])
assert X.shape ==  (52, 316310)
#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape ==  (39, 316310)



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

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mesh(cor, tri, mask, calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)
assert np.allclose(Atv_.get_singular_values(0), 8.999, rtol=1e-03, atol=1e-03)



#############################################################################
# MULM
mask_arr = np.load(os.path.join(OUTPUT, "mask.npy"))
X = np.load(os.path.join(OUTPUT, "X.npy"))
y = np.load(os.path.join(OUTPUT, "y.npy"))
Z = X[:, :2]
Y = X[: , 2:]
assert np.sum(mask_arr) == Y.shape[1]



DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = Z[:, 0]  # age
DesignMat[:, 2] = Z[:, 1]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit()
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0], pval=True)
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","pvals","pvals.npz"),pvals[0])
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","tvals","tvals.npz"),tvals[0])
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","log10pvals","log10pvals.npz"),-np.log10(pvals[0]))


create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/pvals",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/pvals/pvals.npz",
                                   penalty_start = 2,
                                   threshold = False)

create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/tvals",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/tvals/tvals.npz",
                                   penalty_start = 2,
                                   threshold = False)

create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/log10pvals",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/log10pvals/log10pvals.npz",
                                   penalty_start = 2,
                                   threshold = False)


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0]]),nperms=nperms,two_tailed=True)
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","pvals_corr","pvals_corrected_perm.npz"),pvals_perm[0])
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","tvals_corr","tvals_corrected_perm.npz"),tvals_perm[0])
np.savez(os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","univariate_analysis","log10pvals_corr","log10pvals_corrected_perm.npz"),-np.log10(pvals_perm[0]))


create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/pvals_corr",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/pvals_corr/pvals_corrected_perm.npz",
                                   penalty_start = 2,
                                   threshold = False)

create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/tvals_corr",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/tvals_corr/tvals_corrected_perm.npz",
                                   penalty_start = 2,
                                   threshold = False)

create_texture.create_texture_file(OUTPUT="/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/log10pvals_corr",
                                   TEMPLATE_PATH ="/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_template",
                                   MASK_PATH = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/data/mask.npy",
                                   beta_path = "/neurospin/brainomics/2016_icaar-eugei/results/Freesurfer/ICAAR/univariate_analysis/log10pvals_corr/log10pvals_corrected_perm.npz",
                                   penalty_start = 2,
                                   threshold = False)