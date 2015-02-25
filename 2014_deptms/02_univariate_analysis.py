# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:58:54 2014

@author: cp243490

Univariate analysis for the whole brain using MULM module.
Pvalue are corrected for multiple comparisons with test maxT on the whole
brain and one each each ROI.

INPUTS
for one of the three modalities (MRI, PET, MRI+PET)
    - implicit masks:
        /neurospin/brainomics/2014_deptms/datasets/*/mask_brain.nii
    - X whole brain:
        /neurospin/brainomics/2014_deptms/datasets/*/X_*.npy
    - y (response to treatment):
        /neurospin/brainomics/2014_deptms/datasets/*/y.npy

OUTPUTS
for the defined modality
- mulm results :
    /neurospin/brainomics/2014_deptms/results_univariate/*/
        t_stat_rep_min_norep*_brain.nii.gz
        pval-quantile_rep_min_norep*_brain.nii.gz
        pval_rep_min_norep*_brain.nii.gz
        pval-log10_rep_min_norep*_brain.nii.gz
        ptmax_rep_min_norep_*.nii.gz
        ptmax-log10_rep_min_norep_*.nii.gz
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import brainomics.image_atlas

BASE_PATH = "/neurospin/brainomics/2014_deptms"

MODALITIES = ["MRI", "PET", "MRI+PET"]

DATASET_PATH = os.path.join(BASE_PATH,    "datasets")
BASE_DATA_PATH = os.path.join(BASE_PATH,    "base_data")

INPUT_ROIS_CSV = os.path.join(BASE_DATA_PATH,    "ROI_labels.csv")

OUTPUT_UNIVARIATE = os.path.join(BASE_PATH, "results_univariate")


if not os.path.exists(OUTPUT_UNIVARIATE):
    os.makedirs(OUTPUT_UNIVARIATE)

#############################################################################
## Read ROIs csv
rois = []
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["atlas_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if ((not roi_name in rois)
              and (roi_name != "Maskdep-sub")
              and (roi_name != "Maskdep-cort")):
            print "ROI: ", roi_name
            rois.append(roi_name)
print "\n"

#############################################################################
## Univariate analysis for all the Modalities
for MODALITY in MODALITIES:

    print "Modality: ", MODALITY
    DATA_MODALITY_PATH = os.path.join(DATASET_PATH, MODALITY)
    OUTPUT_MODALITY = os.path.join(OUTPUT_UNIVARIATE, MODALITY)
    if not os.path.exists(OUTPUT_MODALITY):
        os.makedirs(OUTPUT_MODALITY)
    ROIS_STAT_PATH = os.path.join(OUTPUT_MODALITY, "tstat_ROI")
    if not os.path.exists(ROIS_STAT_PATH):
        os.makedirs(ROIS_STAT_PATH)
    #########################################################################
    ## MULM
    #########################################################################

    ###############
    # whole brain
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        mask_ima = nib.load(os.path.join(DATA_MODALITY_PATH,
                                         "mask_brain.nii"))
        mask_arr = mask_ima.get_data() != 0
    elif MODALITY == "MRI+PET":
        mask_ima = nib.load(os.path.join(DATA_MODALITY_PATH,
                                         "mask_brain.nii"))
        mask_arr = mask_ima.get_data() != 0
        #concatenate 2 masks 1 for MRI image, the other one for PET image
        mask_arr = np.vstack((mask_arr, mask_arr))
    X = np.load(os.path.join(DATA_MODALITY_PATH,
                             "X_" + MODALITY + "_brain.npy"))
    y = np.load(os.path.join(DATA_MODALITY_PATH,
                             "y.npy"))
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
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                       "t_stat_rep_min_norep_" + MODALITY + "_brain.nii.gz"))

    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "t_stat_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "t_stat_rep_min_norep_PET_brain.nii.gz"))

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

    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "p-quantile_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p-quantile_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p-quantile_rep_min_norep_PET_brain.nii.gz"))

    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = pvals[0]
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "p_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p_rep_min_norep_PET_brain.nii.gz"))

    log10_pvals = -np.log10(pvals)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = log10_pvals[0]
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "p-log10_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p-log10_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "p-log10_rep_min_norep_PET_brain.nii.gz"))

    ########################################
    ## Univariate Analysis with permutations

    #Univariate analysis with empirical pvalues corrected by permutations
    # Permutation procedure to correct the pvalue: maxT

    nperms = 1000

    ##############
    ## whole brain

    muols = MUOLS(Y=Y, X=DesignMat)
    muols.fit()

    tvals, pvals_perm, _ = muols.t_test_maxT(contrasts=contrasts,
                                        nperms=nperms,
                                        two_tailed=True)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = tvals[0]
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                       "t_stat_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "t_stat_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "t_stat_rep_min_norep_PET_brain.nii.gz"))

    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = pvals_perm[0]
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                 "ptmax_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "ptmax_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "ptmax_rep_min_norep_PET_brain.nii.gz"))

    log10_pvals_perm = -np.log10(pvals_perm)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = log10_pvals_perm[0]
    if (MODALITY == "MRI") or (MODALITY == "PET"):
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                 "ptmax-log10_rep_min_norep_" + MODALITY + "_brain.nii.gz"))
    if MODALITY == "MRI+PET":
        arr_MRI = arr[:(arr.shape[0] / 2), :, :]
        arr_PET = arr[(arr.shape[0] / 2):, :, :]
        out_im_MRI = nib.Nifti1Image(arr_MRI, affine=mask_ima.get_affine())
        out_im_MRI.to_filename(os.path.join(OUTPUT_MODALITY,
                           "ptmax-log10_rep_min_norep_MRI_brain.nii.gz"))
        out_im_PET = nib.Nifti1Image(arr_PET, affine=mask_ima.get_affine())
        out_im_PET.to_filename(os.path.join(OUTPUT_MODALITY,
                           "ptmax-log10_rep_min_norep_PET_brain.nii.gz"))

#    ################
#    ## For each ROI

    # Univariate analysis with permutation
    for roi in rois:
        print "ROI: ", roi
        ROI_PATH = os.path.join(ROIS_STAT_PATH, MODALITY + '_' + roi)
        if not os.path.exists(ROI_PATH):
            os.makedirs(ROI_PATH)
        if (MODALITY == "MRI") or (MODALITY == "PET"):
            mask_ima = nib.load(os.path.join(DATA_MODALITY_PATH,
                               "mask_" + roi + ".nii"))
            mask_arr = mask_ima.get_data() != 0
        elif MODALITY == "MRI+PET":
            mask_ima_mri = nib.load(os.path.join(DATA_MODALITY_PATH,
                               "mask_" + roi + ".nii"))
            mask_arr_mri = mask_ima_mri.get_data() != 0
            mask_ima_pet = nib.load(os.path.join(DATA_MODALITY_PATH,
                               "mask_" + roi + ".nii"))
            mask_arr_pet = mask_ima_pet.get_data() != 0
            mask_arr = np.vstack((mask_arr_mri, mask_arr_pet))
        X = np.load(os.path.join(DATA_MODALITY_PATH,
                            "X_" + MODALITY + "_" + roi + ".npy"))
        y = np.load(os.path.join(DATA_MODALITY_PATH, "y.npy"))
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
        arr_tval = np.zeros(mask_arr.shape)
        arr_tval[mask_arr] = tvals[0]
        arr_pval = np.zeros(mask_arr.shape)
        arr_pval[mask_arr] = pvals_perm[0]
        log10_pvals_perm = -np.log10(pvals_perm)
        arr_logpval = np.zeros(mask_arr.shape)
        arr_logpval[mask_arr] = log10_pvals_perm[0]
        if (MODALITY == "MRI") or (MODALITY == "PET"):
            out_im = nib.Nifti1Image(arr_tval, affine=mask_ima.get_affine())
            out_im.to_filename(os.path.join(ROI_PATH,
                 "t_stat_rep_min_norep_" + MODALITY + "_" + roi + ".nii.gz"))
            out_im = nib.Nifti1Image(arr_pval, affine=mask_ima.get_affine())
            out_im.to_filename(os.path.join(ROI_PATH,
                "ptmax_rep_min_norep_" + MODALITY + "_" + roi + ".nii.gz"))
            out_im = nib.Nifti1Image(arr_logpval, affine=mask_ima.get_affine())
            out_im.to_filename(os.path.join(ROI_PATH,
            "ptmax-log10_rep_min_norep_" + MODALITY + "_" + roi + ".nii.gz"))

        elif MODALITY == "MRI+PET":
            arr_tval_MRI = arr_tval[:(arr_tval.shape[0] / 2), :, :]
            arr_tval_PET = arr_tval[(arr_tval.shape[0] / 2):, :, :]
            out_im_MRI = nib.Nifti1Image(arr_tval_MRI,
                                         affine=mask_ima.get_affine())
            out_im_MRI.to_filename(os.path.join(ROI_PATH,
                          "t_stat_rep_min_norep_MRI_" + roi + ".nii.gz"))
            out_im_PET = nib.Nifti1Image(arr_tval_PET,
                                         affine=mask_ima.get_affine())
            out_im_PET.to_filename(os.path.join(ROI_PATH,
                          "t_stat_rep_min_norep_PET_" + roi + ".nii.gz"))

            arr_pval_MRI = arr_pval[:(arr_pval.shape[0] / 2), :, :]
            arr_pval_PET = arr_pval[(arr_pval.shape[0] / 2):, :, :]
            out_im_MRI = nib.Nifti1Image(arr_pval_MRI,
                                         affine=mask_ima.get_affine())
            out_im_MRI.to_filename(os.path.join(ROI_PATH,
                          "ptmax_rep_min_norep_MRI_" + roi + ".nii.gz"))
            out_im_PET = nib.Nifti1Image(arr_pval_PET,
                                         affine=mask_ima.get_affine())
            out_im_PET.to_filename(os.path.join(ROI_PATH,
                         "ptmax_rep_min_norep_PET_" + roi + ".nii.gz"))

            arr_logpval_MRI = arr_logpval[:(arr_logpval.shape[0] / 2), :, :]
            arr_logpval_PET = arr_logpval[(arr_logpval.shape[0] / 2):, :, :]
            out_im_MRI = nib.Nifti1Image(arr_logpval_MRI,
                                         affine=mask_ima.get_affine())
            out_im_MRI.to_filename(os.path.join(ROI_PATH,
                         "ptmax-log10_rep_min_norep_MRI_" + roi + ".nii.gz"))
            out_im_PET = nib.Nifti1Image(arr_logpval_PET,
                                         affine=mask_ima.get_affine())
            out_im_PET.to_filename(os.path.join(ROI_PATH,
                         "ptmax-log10_rep_min_norep_PET_" + roi + ".nii.gz"))