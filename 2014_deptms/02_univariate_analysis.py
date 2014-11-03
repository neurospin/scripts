# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:58:54 2014

@author: cp243490

Univariate analysis for the whole brain using MULM module

INPUTS
for one of the three modalities (MRI, PET, MRI+PET)
    - implicit masks:
        /neurospin/brainomics/2014_deptms/datasets/*/mask*_wb.nii
    - X whole brain:
        /neurospin/brainomics/2014_deptms/datasets/*/X*_wb.npy
    - y (response to treatment):
        /neurospin/brainomics/2014_deptms/datasets/*/y.npy

OUTPUTS
for the defined modality
- mulm results :
    /neurospin/brainomics/2014_deptms/results_univariate/*/
        t_stat_rep_min_norep*_wb.nii.gz
        pval-quantile_rep_min_norep*_wb.nii.gz
        pval_rep_min_norep*_wb.nii.gz
        pval-log10_rep_min_norep*_wb.nii.gz
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

OUTPUT_UNIVARIATE = os.path.join(BASE_PATH,    "results_univariate")

if not os.path.exists(OUTPUT_UNIVARIATE):
    os.makedirs(OUTPUT_UNIVARIATE)

#############################################################################
## Read ROIs csv
rois = []
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["label_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in rois:
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

    #########################################################################
    ## MULM
    #########################################################################

    ###############
    ## whole brain

    mask_ima = nib.load(os.path.join(DATA_MODALITY_PATH,
                                     "mask_" + MODALITY + "_wb.nii"))
    mask_arr = mask_ima.get_data() != 0
    X = np.load(os.path.join(DATA_MODALITY_PATH,
                             "X_" + MODALITY + "_wb.npy"))
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
    DesignMat[:, 3] = Z[:, 2]  # sex

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
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                       "t_stat_rep_min_norep_" + MODALITY + "_wb.nii.gz"))

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
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "pval-quantile_rep_min_norep_" + MODALITY + "_wb.nii.gz"))

    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = pvals[0]
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "pval_rep_min_norep_" + MODALITY + "_wb.nii.gz"))
    log10_pvals = -np.log10(pvals)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = log10_pvals[0]
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                  "pval-log10_rep_min_norep_" + MODALITY + "_wb.nii.gz"))

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
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                       "t_stat_rep_min_norep_" + MODALITY + "_wb.nii.gz"))

    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = pvals_perm[0]
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                 "pval-perm_rep_min_norep_" + MODALITY + "_wb.nii.gz"))
    log10_pvals_perm = -np.log10(pvals_perm)
    arr = np.zeros(mask_arr.shape)
    arr[mask_arr] = log10_pvals_perm[0]
    out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
    out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                 "pval-perm-log10_rep_min_norep_" + MODALITY + "_wb.nii.gz"))

    ################
    ## For each ROI

    # Univariate analysis with permutation
    for roi in rois:
        print "ROI: ", roi
        mask_ima = nib.load(os.path.join(DATA_MODALITY_PATH,
                           "mask_" + MODALITY + "_" + roi + ".nii"))
        mask_arr = mask_ima.get_data() != 0
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
        arr = np.zeros(mask_arr.shape)
        arr[mask_arr] = pvals_perm[0]
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
          "pval-perm_rep_min_norep_" + MODALITY + "_" + roi + ".nii.gz"))
        log10_pvals_perm = -np.log10(pvals_perm)
        arr = np.zeros(mask_arr.shape)
        arr[mask_arr] = log10_pvals_perm[0]
        out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
          "pval-perm-log10_rep_min_norep_" + MODALITY + "_" + roi + ".nii.gz"))