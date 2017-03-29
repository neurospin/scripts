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

BASE_PATH="/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI"

#BIOMARKERS FROM Whole brain analysis
################################################################################
MASK_PATH = os.path.join(BASE_PATH,"data","mask_whole_brain.nii")
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 4


nilearn.plotting.plot_glass_brain(MASK_PATH,plot_abs=False,colorbar= True,vmin=0,vmax=5)


#
#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
whole_brain/svm/vbm_svm_icaar_whole_brain/model_selectionCV/all/all/100"
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
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
whole_brain/enettv/vbm_enettv_icaar_whole_brain/model_selectionCV/refit/refit/1.0_0.09_0.21_0.7"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
####################################################################################
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.00001,vmax = 0.000001)


#BIOMARKERS FROM NUDAST ANALYSIS
################################################################################
MASK_PATH = os.path.join(BASE_PATH,"data","mask_biomarkers.nii")
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 4


nilearn.plotting.plot_glass_brain(MASK_PATH,plot_abs=False,colorbar= True,vmin=0,vmax=5)


#
#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
masked_brain/svm/vbm_svm_icaar_biomarkers/model_selectionCV/all/all/1e-05"
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
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
masked_brain/enettv/vbm_enettv_icaar_biomarkers/model_selectionCV/refit/refit/0.1_0.63_0.07_0.3"
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
penalty_start = 4


nilearn.plotting.plot_glass_brain(MASK_PATH,plot_abs=False,colorbar= True,vmin=0,vmax=2)



#
#SVM
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
masked_brain/svm/vbm_svm_icaar_atlas_subcort/model_selectionCV/all/all/0.0001"
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
WD = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR+EUGEI/results/\
masked_brain/enettv/vbm_enettv_icaar_atlas_subcort/model_selectionCV/refit/refit/0.1_0.0_0.7_0.3"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.01,vmax=0.01)

