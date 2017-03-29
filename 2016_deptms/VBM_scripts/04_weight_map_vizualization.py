# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:21:27 2016

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
POPULATION_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/population.csv"


##################################################################################
babel_mask  = nibabel.load("/neurospin/brainomics/2016_deptms/analysis/VBM/data/mask.nii")
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################


#
#Enet-TV
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/0.1_0.06_0.54_0.4"

WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/model_selectionCV/all/all/0.1_0.01_0.09_0.9"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.0015,vmin = -0.0015)


#Enet-TV
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/0.1_0.12_0.28_0.6"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.001,vmin = -0.001)


#Enet-TV
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/0.1_0.08_0.72_0.2"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.001,vmin = -0.001)


#svm 
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/svm/model_selection_5folds/1"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)






WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/enettv/model_selection_5folds/0.1_0.0_0.1_0.9"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.001,vmin = -0.001)











