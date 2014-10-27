# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:58:54 2014

@author: cp243490

Univariate analysis for the whole brain using MULM module

INPUTS
for one of the three modalities (MRI, PET, MRI+PET)
    - implicit masks:
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/mask*_wb.nii
    - X whole brain:
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/X*_wb.npy
    - y (response to treatment):
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/y.npy

OUTPUTS
for the defined modality
- mulm results :
    /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/
        t_stat_rep_min_norep*_wb.nii.gz
        pval-quantile_rep_min_norep*_wb.nii.gz
        pval_rep_min_norep*_wb.nii.gz
        pval-log10_rep_min_norep*_wb.nii.gz
"""

import os
import numpy as np
import nibabel as nib
import brainomics.image_atlas

MODALITY = "MRI+PET"        # modalities: {"MRI", "PET", "MRI+PET"}

print "Modality: ", MODALITY

BASE_PATH = "/neurospin/brainomics/2014_deptms"

MODALITY_PATH = os.path.join(BASE_PATH,          MODALITY)

#############################################################################
# MULM
# whole brain

mask_ima = nib.load(os.path.join(MODALITY_PATH,
                                 "mask_" + MODALITY + "_wb.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(MODALITY_PATH,
                         "X_" + MODALITY + "_wb.npy"))
y = np.load(os.path.join(MODALITY_PATH,
                         "y.npy"))
Z = X[:, :3]
Y = X[:, 3:]
assert np.sum(mask_arr) == Y.shape[1]

DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))  # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


from mulm import MUOLSStatsCoefficients

# whole brain
muols = MUOLSStatsCoefficients()
muols.fit(X=DesignMat, Y=Y)

tvals, pvals, dfs = muols.stats_t_coefficients(X=DesignMat, Y=Y,
                                               contrast=[1, 0, 0, 0],
                                               pval=True)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "t_stat_rep_min_norep_" +
                                MODALITY + "_wb.nii.gz"))

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
arr[mask_arr] = val
arr = brainomics.image_atlas.smooth_labels(arr, size=(3, 3, 3))
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "pval-quantile_rep_min_norep_" +
                                MODALITY + "_wb.nii.gz"))

arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "pval_rep_min_norep_" +
                                MODALITY + "_wb.nii.gz"))
log10_pvals = -np.log10(pvals)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "pval-log10_rep_min_norep_" +
                                MODALITY + "_wb.nii.gz"))