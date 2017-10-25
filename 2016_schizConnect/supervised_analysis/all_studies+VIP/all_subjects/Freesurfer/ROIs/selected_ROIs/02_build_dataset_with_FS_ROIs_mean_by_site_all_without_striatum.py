#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:40:21 2017

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
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_selected_ROIs"
INPUT_SCZCO_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
INPUT_VIP_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"


# Create Volume dataset
#######################################################################################################
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
vol_sczco = pd.read_csv(INPUT_SCZCO_VOLUME,sep='\t')
vol_vip = pd.read_csv(INPUT_VIP_VOLUME,sep='\t')
vol_all = vol_sczco.append(vol_vip, ignore_index=True)

fs_path_vip = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/data/original_data/passed_QC/'
fs_path_sczco = "/neurospin/abide/schizConnect/processed/freesurfer/passed_QC"


site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy")
#################################################
i=0
for p in pop["mri_path_lh"]:
    print(os.path.basename(p)[:-7])
    pop["id"][i] = os.path.basename(p)[:-7]
    i=i+1
####################################################
i=0
vol_all["id"] = vol_all["Measure:volume"]
for p in vol_all["Measure:volume"]:
    print(os.path.basename(p))
    vol_all["id"][i] = os.path.basename(p)
    i=i+1

table = pop.merge(vol_all, on="id")
y = np.asarray(table["dx_num"])
cov = table[["age",'sex_num']]
cov = np.asarray(cov)
stats = np.asarray(table)


assert table.shape == (567, 79)

ROIs =  (["age",'sex_num',
       'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
       'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
       'Left-Pallidum','Brain-Stem',
       'Left-Hippocampus', 'Left-Amygdala', 'CSF', 'Left-Accumbens-area',
       'Left-VentralDC', 'Left-choroid-plexus',
       'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
       'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
       'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
       'Right-Accumbens-area', 'Right-VentralDC',
       'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf',
       'lhCortexVol', 'rhCortexVol', 'CortexVol',
       'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
       'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
       'SupraTentorialVol', 'SupraTentorialVolNotVent',
       'SupraTentorialVolNotVentVox', 'MaskVol', 'BrainSegVol-to-eTIV', 'EstimatedTotalIntraCranialVol'])

table = table[ROIs]
assert table.shape == (567, 49)
features = table.get_values()
stats = np.asarray(table)
Xv = stats.astype(float)
assert Xv.shape == (567,49)

Xv[site==1,:] = Xv[site==1,:] - Xv[site==1,:].mean(axis=0)
Xv[site==2,:] = Xv[site==2,:] - Xv[site==2,:].mean(axis=0)
Xv[site==3,:] = Xv[site==3,:] - Xv[site==3,:].mean(axis=0)
Xv[site==4,:] = Xv[site==4,:] - Xv[site==4,:].mean(axis=0)


np.save(os.path.join(OUTPUT_DATA,"Xrois_volumes_mean_centered_by_site+cov.npy"),Xv)
np.save(os.path.join(OUTPUT_DATA,"y.npy"),y)
np.save(os.path.join(OUTPUT_DATA,"features.npy"),features)

