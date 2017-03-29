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

INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/data/mask.nii'              



babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
#############################################################################
#SCZ + CONTROLS
#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/\
pcatv_all/5_folds_NMoprhCH_all/results/0/struct_pca_0.1_0.5_0.1"
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
#SCZ  only
#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/\
pcatv_scz/5_folds_NMoprhCH_scz/results/0/struct_pca_0.1_0.8_0.5"
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
#controls
#############################################################################
WD = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/results/\
pcatv_controls/5_folds_NMoprhCH_controls/results/0/struct_pca_0.1_0.8_0.5"
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