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

BASE_PATH="/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM"
POPULATION_CSV = os.path.join(BASE_PATH,"population.csv")
MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/enetall_NUSDAST_all/mask.nii"


##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
penalty_start = 2

#
#SVM
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/svm/svm_model_selection_5folds_NUDAST/model_selectionCV/all/all/1e-05"
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
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/enetall_NUSDAST_all/5cv/refit/refit/enettv_0.01_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.0004, vmax = 0.0004)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",vmax = 0.005)
