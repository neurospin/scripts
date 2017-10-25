#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:24:38 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image

BASE_PATH="/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/"
MASK_PATH = os.path.join(BASE_PATH,"data","mask.nii")

##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################


penalty_start = 2
#SVM
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/COBRE/enetall/5cv/refit/refit/enettv_0.1_0.1_0.8"

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/NMORPH/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8"

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/NUSDAST/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8"


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/VIP/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8"

beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",cut_coords=(-1,-13,14),vmax=0.1)



beta1= np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/COBRE/enetall/5cv/refit/refit/enetgn_0.01_0.5_0.8/beta.npz")['arr_0'][penalty_start:]

beta2= np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/NMORPH/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8/beta.npz")['arr_0'][penalty_start:]

beta3= np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/NUSDAST/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8/beta.npz")['arr_0'][penalty_start:]

beta4= np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/by_site/VIP/enetall/5cv/refit/refit/enettv_0.01_0.1_0.8/beta.npz")['arr_0'][penalty_start:]

# Correlation

beta1_t,t = array_utils.arr_threshold_from_norm2_ratio(beta1, .99)
beta2_t,t = array_utils.arr_threshold_from_norm2_ratio(beta2, .99)
beta3_t,t = array_utils.arr_threshold_from_norm2_ratio(beta3, .99)
beta4_t,t = array_utils.arr_threshold_from_norm2_ratio(beta4, .99)
betas_t = np.hstack((beta1_t,beta2_t,beta3_t,beta4_t))


betas = np.hstack((beta1,beta2,beta3,beta4))

R = np.corrcoef(betas.T)
R = R[np.triu_indices_from(R, 1)]
R.mean()


