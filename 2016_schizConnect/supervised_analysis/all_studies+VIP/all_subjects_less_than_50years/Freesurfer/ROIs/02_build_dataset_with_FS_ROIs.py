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


INPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/population_50yo.csv"
INPUT_RH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/results/ROIs_analysis/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_LH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/results/ROIs_analysis/freesurfer_stats/aparc_thickness_lh_all.csv"
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs"

INPUT_SCZCO_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
INPUT_VIP_VOLUME = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"


# Create Volume dataset
pop = pd.read_csv(INPUT_CSV)
vol_sczco = pd.read_csv(INPUT_SCZCO_VOLUME,sep='\t')
vol_vip = pd.read_csv(INPUT_VIP_VOLUME,sep='\t')
vol_all = vol_sczco.append(vol_vip, ignore_index=True)

fs_path_vip = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/data/original_data/passed_QC/'
fs_path_sczco = "/neurospin/abide/schizConnect/processed/freesurfer/passed_QC"

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
cov = table[["age",'sex_num','site_num']]
cov = np.asarray(cov)
stats = np.asarray(table)
Xv = stats[:,13:].astype(float)
features = table.keys()[13:]
features =features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs/Xrois_volumes.npy",Xv)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs/features.npy",features)

X = np.hstack([cov, Xv])
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs/Xrois_vol+cov.npy",X)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs/y.npy",y)

featuresAndCov = (["age",'sex_num','site_num','Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
       'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
       'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
       'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem',
       'Left-Hippocampus', 'Left-Amygdala', 'CSF', 'Left-Accumbens-area',
       'Left-VentralDC', 'Left-vessel', 'Left-choroid-plexus',
       'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
       'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
       'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
       'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
       'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel',
       'Right-choroid-plexus', '5th-Ventricle', 'WM-hypointensities',
       'Left-WM-hypointensities', 'Right-WM-hypointensities',
       'non-WM-hypointensities', 'Left-non-WM-hypointensities',
       'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf',
       'lhCortexVol', 'rhCortexVol', 'CortexVol',
       'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
       'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
       'SupraTentorialVol', 'SupraTentorialVolNotVent',
       'SupraTentorialVolNotVentVox', 'MaskVol', 'BrainSegVol-to-eTIV',
       'MaskVol-to-eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles',
       'SurfaceHoles', 'EstimatedTotalIntraCranialVol'])

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects_less_than_50years/data/data_ROIs/featuresAndCov.npy",featuresAndCov)



