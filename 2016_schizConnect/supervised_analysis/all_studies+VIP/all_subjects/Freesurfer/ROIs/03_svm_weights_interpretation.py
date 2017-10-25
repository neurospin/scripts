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
####################################################
pop = pd.read_csv(INPUT_CSV)
vol_sczco = pd.read_csv(INPUT_SCZCO_VOLUME,sep='

vol_vip = pd.read_csv(INPUT_VIP_VOLUME,sep='\t')
vol_all = vol_sczco.append(vol_vip, ignore_index=True)


weights = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
Freesurfer/all_subjects/results/ROIs_analysis/with_covariates/svm_5_folds/model_selectionCV/all/all/0.001/beta.npz")['arr_0']


features = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/featuresAndCov.npy")


#remove age, sex and site for plots
weights = weights[0,3:]
features = features[3:]
import matplotlib.pyplot as plt

#plt.plot(np.arange(69),weights[0,:],'o',label = features)
#
#plt.bar(np.arange(69),weights[0,:],label = features)
#

order = np.argsort(np.abs(weights[:]))[::-1]

plt.bar(np.arange(20),weights[order[:20]])
plt.xticks(np.arange(20),  features[order[:20]],fontsize =12)
plt.xticks(rotation="vertical")
plt.ylabel("Feature weights")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/weights/svm_weights_top20.png")



plt.barh(np.arange(20),weights[order[:20]])
plt.set_yticks(np.arange(20))
plt.set_yticklabels(features[order[:20]])


plt.xticks(np.arange(20),  features[order[:20]],fontsize =12)
plt.xticks(rotation="vertical")
plt.ylabel("Feature weights")


plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
# Example data
features_names = features[order[:20]]
y_pos = np.arange(len(features_names))
performance = weights[order[:20]]
ax.barh(y_pos, performance, color='b',alpha = 0.6,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_names,fontsize =16)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature weights')
ax.axis('tight')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/svm_weights_top20.png")









plt.bar(np.arange(69),weights[0,:])
plt.xticks(np.arange(69),  features,fontsize =4)
plt.xticks(rotation="vertical")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/weights/svm_weights.png")


plt.bar(np.arange(69),np.sort(weights[0,:]))
plt.xticks(np.arange(69),  features,fontsize =4)
plt.xticks(rotation="vertical")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/weights/svm_weights.png")



plt.bar(np.arange(34),weights[0,:34])
plt.xticks(np.arange(34),  features[:34], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")


plt.bar(np.arange(35),weights[0,34:])
plt.xticks(np.arange(35),  features[34:], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")

# Create thickness dataset
####################################################
weights = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
Freesurfer/all_subjects/results/ROIs_analysis/thickness/svm_5_folds/model_selectionCV/all/all/0.001/beta.npz")['arr_0']


features = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/featureThksAndCov.npy")

import matplotlib.pyplot as plt

plt.bar(np.arange(73),weights[0,:])
plt.xticks(np.arange(73),  features, rotation='vertical',fontsize =6)


plt.bar(np.arange(34),weights[0,:34])
plt.xticks(np.arange(34),  features[:34], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")


plt.bar(np.arange(39),weights[0,34:])
plt.xticks(np.arange(39),  features[34:], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")


# Create thickness+ vol datase
####################################################
weights = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/\
results/ROIs_analysis/thick+vol/svm_5_folds/model_selectionCV/all/all/0.001/beta.npz")['arr_0']


features = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/features_thick+vol+cov.npy")

import matplotlib.pyplot as plt

plt.bar(np.arange(139),weights[0,:])
plt.xticks(np.arange(139),  features, rotation='vertical',fontsize =6)


plt.bar(np.arange(45),weights[0,:45])
plt.xticks(np.arange(45),  features[:45], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")


plt.bar(np.arange(45),weights[0,45:90])
plt.xticks(np.arange(45),  features[45:90], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")

plt.bar(np.arange(49),weights[0,90:])
plt.xticks(np.arange(49),  features[90:], rotation='vertical',fontsize = 10)
plt.ylabel("Feature weights")
