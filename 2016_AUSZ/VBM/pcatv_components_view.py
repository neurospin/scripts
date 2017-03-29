# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:21:01 2016

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
################
# Input/Output #
################

INPUT_BASE_DIR = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only'
INPUT_DIR = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV'
INPUT_MASK = '/neurospin/brainomics/2016_AUSZ/results/VBM/data/mask.nii'             
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results.csv")                          
INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"5_folds","config.json")
OUTPUT_DIR = os.path.join(INPUT_BASE_DIR,"components_extracted")



babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()



##############
# Parameters #
##############

N_COMP = 5
EXAMPLE_FOLD = 0
INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,'{fold}','{key}','components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,'{fold}','{key}','X_train_transform.npz')
OUTPUT_COMPONENTS_FILE_FORMAT = os.path.join(OUTPUT_DIR,'{name}.nii')



config = json.load(open(INPUT_CONFIG_FILE))





WD = "/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.1_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
for i in range(comp.shape[1]):
    current_comp = comp[:,i]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = current_comp.ravel()
    out_im = nib.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%d.nii.gz" %i)
    out_im.to_filename(filename)


import nilearn  
from nilearn import plotting
from nilearn import image
filename = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.1_0.1/comp_0.nii.gz'
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)


filename = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.1_0.1/comp_1.nii.gz'
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)



filename = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.1_0.1/comp_2.nii.gz'
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)



filename = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.1_0.1/comp_3.nii.gz'
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

