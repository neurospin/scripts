# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 09:04:18 2014

@author: cp243490

Univariate analysis for the whole brain and each ROI
using the permutation procedure (maxT) to obtain empirically
corrected pvalues

INPUTS
for one of the three modalities (MRI, PET, MRI+PET)
    - implicit masks and mask associated to each ROI:
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/mask*.nii
    - X whole brain and X associated to each ROI:
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/X*.npy
    - y (response to treatment):
        /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/y.npy

OUTPUTS
for the defined modality
- mulm results for the permutation. We obtain for the whole brain and each
    ROI the set of empirical pvalues obtained from the permutation procedure
    (maxT)
    /neurospin/brainomics/2014_deptms/*{MRI, PET, PRI+PET}/
        t_stat_rep_min_norep*_wb.nii.gz (whole brain only)
        pval-perm_rep_min_norep*.nii.gz
        pval-perm-log10_rep_min_norep*.nii.gz

"""
import os
import numpy as np
import pandas as pd
import nibabel as nib

MODALITY = "MRI+PET"  # modalities: {"MRI", "PET", "MRI+PET"}

print "Modality: ", MODALITY

BASE_PATH = "/neurospin/brainomics/2014_deptms"

MODALITY_PATH = os.path.join(BASE_PATH,          MODALITY)

INPUT_ROIS_CSV = os.path.join(BASE_PATH,     "ROI_labels.csv")

#############################################################################
# Read ROIs csv
rois = []
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["label_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in rois:
            print "ROI_name_deptms", roi_name
            rois.append(roi_name)


#############################################################################
# MULM
# Univariate analysis with empirical pvalues correctef by permutations (maxT)
# Permutation procedure to correct the pvalue: maxT
nperms = 1000

# whole brain
mask_ima = nib.load(os.path.join(MODALITY_PATH,
                                 "mask_" + MODALITY + "_wb.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(MODALITY_PATH, "X_" + MODALITY + "_wb.npy"))
y = np.load(os.path.join(MODALITY_PATH, "y.npy"))
Z = X[:, :3]
Y = X[:, 3:]
assert np.sum(mask_arr) == Y.shape[1]

DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))  # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


from mulm import MUOLSStatsCoefficients

muols = MUOLSStatsCoefficients()
muols.fit(X=DesignMat, Y=Y)

tvals, pvals_perm, _ = muols.stats_t_permutations(X=DesignMat,
                                                  Y=Y,
                                                  contrast=[1, 0, 0, 0],
                                                  nperms=nperms)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "t_stat_rep_min_norep_" +
                                MODALITY + "_wb.nii.gz"))

arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals_perm
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "pval-perm_rep_min_norep_" +
                                    MODALITY + "_wb.nii.gz"))
log10_pvals_perm = np.log10(pvals_perm)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals_perm
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(MODALITY_PATH,
                                "pval-perm-log10_rep_min_norep_" +
                                    MODALITY + "_wb.nii.gz"))


# Univariate analysis with permytation for each ROI
for roi in rois:
    print "ROI: ", roi
    mask_ima = nib.load(os.path.join(MODALITY_PATH,
                                     "mask_" + MODALITY + "_" + roi + ".nii"))
    mask_arr = mask_ima.get_data() != 0
    X = np.load(os.path.join(MODALITY_PATH,
                             "X_" + MODALITY + "_" + roi + ".npy"))
    y = np.load(os.path.join(MODALITY_PATH, "y.npy"))
    Z = X[:, :3]
    Y = X[:, 3:]
    assert np.sum(mask_arr) == Y.shape[1]

    # y, intercept, age, sex
    DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
    DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
    DesignMat[:, 1] = 1  # intercept
    DesignMat[:, 2] = Z[:, 1]  # age
    DesignMat[:, 3] = Z[:, 2]  # sex

    muols = MUOLSStatsCoefficients()
    muols.fit(X=DesignMat, Y=Y)

    tvals, pvals_perm, _ = muols.stats_t_permutations(X=DesignMat,
                                                      Y=Y,
                                                      contrast=[1, 0, 0, 0],
                                                      nperms=nperms)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = pvals_perm
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(MODALITY_PATH,
                                    "pval-perm_rep_min_norep_" +
                                        MODALITY + "_" + roi + ".nii.gz"))
    log10_pvals_perm = -np.log10(pvals_perm)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = log10_pvals_perm
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(MODALITY_PATH,
                                    "pval-perm-log10_rep_min_norep_" +
                                        MODALITY + "_" + roi + ".nii.gz"))