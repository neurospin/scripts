#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:47:10 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image


MASK_PATH = "/neurospin/brainomics/2016_AUSZ/september_2017/results/\
VBM/data/data_with_intercept/mask.nii"


##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################

#
#Ridge regression

WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_patients_MAASCtot_intercept_penalty_start/RESULTS/5cv/refit/refit/Ridge_1.0_0.1_0"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(-1,-13,14),vmax=0.0008)

#Enettv
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_patients_MAASCtot_intercept_penalty_start/RESULTS/5cv/refit/refit/enettv_0.001_0.1_0.01"

WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_patients_MAASCtot_intercept_penalty_start/RESULTS/5cv/refit/refit/enettv_1.0_0.1_0.1"

beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = t,\
                               cut_coords=(-1,-13,14))

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(-1,-13,14))

#GraphNet
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_patients_MAASCtot_intercept_penalty_start/RESULTS/5cv/refit/refit/enetgn_1.0_0.5_0.01"

beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = t,\
                               cut_coords=(-1,-13,14))


#Enet regression
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_patients_MAASCtot_intercept_penalty_start/RESULTS/5cv/refit/refit/enet_0.01_0.1_0"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(-1,-13,14))
