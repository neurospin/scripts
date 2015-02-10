# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:33:28 2014

@author: cp243490
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import brainomics.image_atlas

MASKDEP_PATH = "/neurospin/brainomics/2014_deptms/maskdep"

DATASET_PATH = os.path.join(MASKDEP_PATH,    "datasets")

OUTPUT_UNIVARIATE = os.path.join(MASKDEP_PATH,    "results_univariate")

if not os.path.exists(OUTPUT_UNIVARIATE):
    os.makedirs(OUTPUT_UNIVARIATE)

#############################################################################
## Univariate analysis for the brain

########
## MULM
########


mask_ima = nib.load(os.path.join(DATASET_PATH, "mask_brain.nii"))
mask_arr = mask_ima.get_data() != 0

X = np.load(os.path.join(DATASET_PATH, "X_brain.npy"))
y = np.load(os.path.join(DATASET_PATH, "y.npy"))
Z = X[:, :3]
Y = X[:, 3:]
assert np.sum(mask_arr) == Y.shape[1]

# y, intercept, age, sex
DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # gender

##################################
## Standard Univariate Analysis
##################################

# whole brain
from mulm import MUOLS

muols = MUOLS(Y=Y, X=DesignMat)
muols.fit()
contrasts = np.array([[1, 0, 0, 0]])
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
               "t_stat_rep_min_norep_brain.nii.gz"))
thres = .1
m1 = pvals <= thres
m2 = (pvals > thres) & (pvals < (1. - thres))
m3 = pvals >= (1. - thres)
print np.sum(m1), np.sum(m2), np.sum(m3)
arr = np.zeros(mask_arr.shape)
val = np.zeros(pvals.shape, dtype=int)
val[m1] = 1.
val[m2] = 2.
val[m3] = 3.
arr[mask_arr] = val[0]
arr = brainomics.image_atlas.smooth_labels(arr, size=(3, 3, 3))
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
          "p-quantile_rep_min_norep_brain.nii.gz"))


arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
          "p_rep_min_norep__brain.nii.gz"))


log10_pvals = -np.log10(pvals)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
          "p-log10_rep_min_norep_brain.nii.gz"))

########################################
## Univariate Analysis with permutations
########################################

#Univariate analysis with empirical pvalues corrected by permutations
# Permutation procedure to correct the pvalue: maxT

nperms = 1000

#################################
# Correctied pvalue for the brain
muols = MUOLS(Y=Y, X=DesignMat)
muols.fit()

tvals, pvals_perm, _ = muols.t_test_maxT(contrasts=contrasts,
                                    nperms=nperms,
                                    two_tailed=True)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
               "t_stat_rep_min_norep_brain.nii.gz"))

arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
         "ptmax_rep_min_norep_brain.nii.gz"))

log10_pvals_perm = -np.log10(pvals_perm)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
         "ptmax-log10_rep_min_norep_brain.nii.gz"))

###################################################
# Correctied pvalue for maskdep (union of all ROIs)

# dilatated_masks
DILATE_PATH = "dilatation_masks"
mask_ima = nib.load(os.path.join(DATASET_PATH, DILATE_PATH,
                                 "mask_dilatation.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(DATASET_PATH, DILATE_PATH, "X_dilatation.npy"))
y = np.load(os.path.join(DATASET_PATH, DILATE_PATH, "y.npy"))
Z = X[:, :3]
Y = X[:, 3:]
assert np.sum(mask_arr) == Y.shape[1]
# y, intercept, age, sex
DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y, X=DesignMat)
muols.fit()

tvals, pvals_perm, _ = muols.t_test_maxT(contrasts=contrasts,
                                         nperms=nperms,
                                         two_tailed=True)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
                    "ptmax_rep_min_norep_dilatation.nii.gz"))

log10_pvals_perm = -np.log10(pvals_perm)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
                    "ptmax-log10_rep_min_norep_dilatation.nii.gz"))

#dilated_mask_within the brain
DILATE_WB_PATH = "dilatation_within-brain_masks"
mask_ima = nib.load(os.path.join(DATASET_PATH, DILATE_WB_PATH,
                                 "mask_dilatation_within-brain.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(DATASET_PATH, DILATE_WB_PATH,
                         "X_dilatation_within-brain.npy"))
y = np.load(os.path.join(DATASET_PATH, DILATE_WB_PATH, "y.npy"))
Z = X[:, :3]
Y = X[:, 3:]
assert np.sum(mask_arr) == Y.shape[1]
# y, intercept, age, sex
DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex

muols = MUOLS(Y=Y, X=DesignMat)
muols.fit()

tvals, pvals_perm, _ = muols.t_test_maxT(contrasts=contrasts,
                                         nperms=nperms,
                                         two_tailed=True)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
                "ptmax_rep_min_norep_dilatation_within-brain.nii.gz"))

log10_pvals_perm = -np.log10(pvals_perm)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
                "ptmax-log10_rep_min_norep_dilatation_within-brain.nii.gz"))