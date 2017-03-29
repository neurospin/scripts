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
  


#sparse_plot= plt.plot(np.arange(1,4),np.abs(evr_sparse.mean(axis=0))*100,'b-o',markersize=5,label = "Sparse PCA")
#enet_plot= plt.plot(np.arange(1,4),np.abs(evr_enet.mean(axis=0))*100,'g-^',markersize=5,label = "ElasticNet")
#tv_plot= plt.plot(np.arange(1,4),np.abs(evr_tv.mean(axis=0))*100,'r-s',markersize=5,label = "PCA-TV")
pca_plot= plt.plot(np.arange(1,11),evr,'r-s',markersize=5)
plt.xlabel("Component Number")
plt.ylabel("Test Data Explained Variance (%)")
plt.legend(loc= 'upper right')


      