#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:13:41 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn  
from nilearn import plotting
from nilearn import image

BASE_PATH="/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR"







#BIOMARKERS FROM WHOLE BRAIN ANALYSIS
################################################################################
MASK_PATH = os.path.join(BASE_PATH,"data","mask_whole_brain.nii")
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3


#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/\
whole_brain/svm/model_selectionCV/all/all/1e-06"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)


#Enet-TV
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/\
whole_brain/enettv/vbm_enettv_icaar_whole_brain/model_selectionCV/refit/refit/1.0_0.15_0.35_0.5"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

#BIOMARKERS FROM NUDAST ANALYSIS
################################################################################
MASK_PATH = os.path.join(BASE_PATH,"data","mask_biomarkers.nii")
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3

#
#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/\
masked_brain/svm/vbm_svm_icaar_biomarkers/model_selectionCV/all/all/1e-08"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)


#Enet-TV
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/masked_brain/\
enettv/vbm_enettv_icaar_biomarkers_0.01_0.36_0.04_0.6/model_selectionCV/refit/refit/0.1_0.63_0.07_0.3"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.0004, vmax = 0.0004)



#BIOMARKERS FROM SUBCORTICAL ATLAS
################################################################################
MASK_PATH = os.path.join(BASE_PATH,"data","data_atlas_subcort","mask_atlas_sub.nii")
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3


nilearn.plotting.plot_glass_brain(MASK_PATH,plot_abs=False,colorbar= True,vmin=0,vmax=2)



#
#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/\
masked_brain/svm/vbm_svm_icaar_atlas_subcort/model_selectionCV/all/all/0.1"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)


#Enet-TV
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/\
masked_brain/enettv/vbm_enettv_icaar_atlas_subcort/model_selectionCV/refit/refit/0.1_0.72_0.08_0.2"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.0004, vmax = 0.0004)

