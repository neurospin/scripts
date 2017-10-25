#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:30:41 2017

@author: ad247405
"""


import os
import numpy as np
import glob
import pandas as pd
import nibabel as nib
import brainomics.image_atlas
import shutil
import mulm
import sklearn
from  scipy import ndimage
import nibabel

INPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/population.csv"
INPUT_RH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_LH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aparc_thickness_lh_all.csv"
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs"

INPUT_SCZCO_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
INPUT_VIP_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"


# Create Volume dataset
pop = pd.read_csv(INPUT_CSV)
vol_sczco = pd.read_csv(INPUT_SCZCO_VOLUME,sep='\t')
vol_vip = pd.read_csv(INPUT_VIP_VOLUME,sep='\t')
vol_all = vol_sczco.append(vol_vip, ignore_index=True)


weights = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
Freesurfer/all_subjects/results/ROIs_analysis/with_covariates/svm_5_folds/model_selectionCV/all/all/0.001/beta.npz")['arr_0']

np.abs(weights).argmax()

features = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/featuresAndCov.npy")

import matplotlib.pyplot as plt

plt.plot(np.arange(69),weights[0,:],'o',label = features)

plt.bar(np.arange(69),weights[0,:],label = features)

plt.bar(np.arange(69),weights[0,:])
plt.xticks(np.arange(69),  features, rotation='vertical',fontsize =6)


plt.bar(np.arange(34),weights[0,:34])
plt.xticks(np.arange(34),  features[:34], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")


plt.bar(np.arange(35),weights[0,34:])
plt.xticks(np.arange(35),  features[34:], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")
