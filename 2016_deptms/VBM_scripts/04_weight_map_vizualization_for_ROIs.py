#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 11:27:41 2016

@author: ad247405
"""


import os
import numpy as np
import nibabel
import array_utils
import nilearn  
from nilearn import plotting
from nilearn import image

BASE_PATH="/neurospin/brainomics/2016_deptms"





#ROI Hippocampus
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv_ROIs/Roiho-hippo/"
babel_mask  = nibabel.load(os.path.join(WD,"mask.nii"))
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

beta = np.load(os.path.join(WD,"0.1_0.09999999999999998_0.09999999999999998_0.8","beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.01,vmin = -0.01)


#ROI Cingulum Anterior
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv_ROIs/Roiho-cingulumAnt/"
babel_mask  = nibabel.load(os.path.join(WD,"mask.nii"))
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

beta = np.load(os.path.join(WD,"0.1_0.019999999999999997_0.17999999999999997_0.8","beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.005,vmin = -0.005)
