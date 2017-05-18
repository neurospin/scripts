#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:04:33 2016

@author: ad247405
"""

import os
import numpy as np
import nibabel
import matplotlib.pyplot as plt
import array_utils

BASE_PATH = "/neurospin/brainomics/2016_pca_struct/fmri"
OUTPUT = "/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/components_vizu"

##################################################################################
babel_mask  = nibabel.load("/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/mask.nii.gz")
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
N_COMP = 10

#
#Enet-TV
WD = "/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/model_selectionCV/all/all/struct_pca_0.1_0.5_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(OUTPUT,"enettv","comp_%d.nii.gz" %i)
    out_im.to_filename(filename)


    #Enet
WD = "/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/model_selectionCV/all/all/struct_pca_0.1_1e-06_0.5"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(OUTPUT,"enet","comp_%d.nii.gz" %i)
    out_im.to_filename(filename)

    #sparse
WD = "/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/model_selectionCV/all/all/sparse_pca_0.0_0.0_5.0"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(OUTPUT,"sparse","comp_%d.nii.gz" %i)
    out_im.to_filename(filename)

    #PCA
WD = "/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/model_selectionCV/all/all/pca_0.0_0.0_0.0"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(OUTPUT,"pca","comp_%d.nii.gz" %i)
    out_im.to_filename(filename)
evr = np.load(os.path.join(WD,"evr_test.npz"))['arr_0']



    #PCA GraphNet
WD = "/neurospin/brainomics/2016_pca_struct/fmri/2017_GraphNet_fmri/model_selectionCV/all/all/graphNet_pca_0.1_0.5_0.5"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(OUTPUT,"graphNet","comp_%d.nii.gz" %i)
    out_im.to_filename(filename)


#SSPCA Jenatton
WD = "/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton/vizu/"

comp = np.load("/home/ad247405/Desktop/V.npy")

for i in range(3):
    current_comp = comp[:,i]
    current_comp_s = current_comp.reshape(mask_bool.shape)
    out_im = nibabel.Nifti1Image(current_comp_s, affine=babel_mask.get_affine())
    filename = os.path.join(WD ,"comp_%d.nii.gz" %i)
    out_im.to_filename(filename)


beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)






################ PCA TV for controls ##########################################

WD = "/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_controls_only_5x5/model_selectionCV/0/struct_pca_0.1_0.5_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(N_COMP):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%d.nii.gz" %i)
    out_im.to_filename(filename)


