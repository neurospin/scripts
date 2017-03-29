#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:39:33 2017

@author: ad247405
"""


import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil
import mulm
import sklearn
import array_utils


BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM'
OUTPUT = os.path.join(BASE_PATH,"data","data_ROIs")
INPUT_MASK_BIOMARKER = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/\
enettv/VIP_enettv/model_selectionCV/refit/refit/0.01_0.09_0.81_0.1"
ORIGINAL_MASK = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/mask.nii"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/X_scz_only.npy'       

X = np.load(INPUT_DATA_X)
original_mask = nibabel.load(ORIGINAL_MASK).get_data()

beta = nibabel.load(os.path.join(INPUT_MASK_BIOMARKER,"weight_map.nii.gz")).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
biomarker_mask = beta_t !=0
assert biomarker_mask.sum() == 78229



mask  = original_mask[np.logical_not(biomarker_mask)] = 0
assert np.sum(mask != 0) == 41625


np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_ROIs/X_masked.npy,X)

  

