# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:12:45 2016

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:46:54 2016

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel
import pandas as pd
import nibabel as nib
import json
import nilearn
from nilearn import plotting
from nilearn import image
import array_utils

################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv'
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"5_folds_NUDAST","results")
INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/mask.nii'



babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
#############################################################################
#SCZ only
#############################################################################
#WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/\
#pcatv_10comp/5_folds_NUDAST_10comp/results/0/struct_pca_0.1_0.1_0.1"
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/\
pcatv_scz/5_folds_NUDAST_10comp/results/0/struct_pca_0.1_0.1_0.1"

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1"

comp = np.load(os.path.join(WD,"components.npz"))['arr_0']



N_COMP =10

for i in range(comp.shape[1]):
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] =comp[:,i]
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%s.nii.gz") % (i)
    out_im.to_filename(filename)
    comp_data = nibabel.load(filename).get_data()
    comp_t,t = array_utils.arr_threshold_from_norm2_ratio(comp_data, .99)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
    print (i)
    print (t)


#############################################################################
#CONTROLS only
#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/\
pcatv_controls/5_folds_NUDAST_controls/results/0/struct_pca_0.1_0.1_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']


N_COMP =10

for i in range(comp.shape[1]):
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] =comp[:,i]
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%s.nii.gz") % (i)
    out_im.to_filename(filename)
    comp_data = nibabel.load(filename).get_data()
    comp_t,t = array_utils.arr_threshold_from_norm2_ratio(comp_data, .99)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
    print (i)
    print (t)



#############################################################################
#ALL SUBEJCTS: scz + controls
#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/\
pcatv_all/5_folds_NUDAST_all/results/0/struct_pca_0.1_0.5_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']


N_COMP =10

for i in range(comp.shape[1]):
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] =comp[:,i]
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%s.nii.gz") % (i)
    out_im.to_filename(filename)
    comp_data = nibabel.load(filename).get_data()
    comp_t,t = array_utils.arr_threshold_from_norm2_ratio(comp_data, .99)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
    print (i)
    print (t)