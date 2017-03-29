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

BASE_PATH="/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects_less_than_30years"
POPULATION_CSV = os.path.join(BASE_PATH,"population.csv")
MASK_PATH = os.path.join(BASE_PATH,"data","mask.nii")


##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
penalty_start = 3

#
#SVM
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects_less_than_30years/results/svm/svm_model_selection_5folds_all_30yo/model_selectionCV/all/all/0.0001"
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
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/results/enettv/3mm_enettv_model_selection_5folds_all_subjects/model_selectionCV/refit/refit/0.1_0.18_0.02_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin = -0.0002, vmax = 0.0002)

