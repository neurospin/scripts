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
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/mean_centered_by_site_controls"
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
Xv = stats[:,13:].astype(float)
features = table.keys()[13:]
features = features.get_values()

Xv_controls = Xv[y==0,:]
site_controls = site[y==0]
Xv[site==1,:] = Xv[site==1,:] - Xv_controls[site_controls==1,:].mean(axis=0)
Xv[site==2,:] = Xv[site==2,:] - Xv_controls[site_controls==2,:].mean(axis=0)
Xv[site==3,:] = Xv[site==3,:] - Xv_controls[site_controls==3,:].mean(axis=0)
Xv[site==4,:] = Xv[site==4,:] - Xv_controls[site_controls==4,:].mean(axis=0)

np.save(os.path.join(OUTPUT_DATA,"Xrois_volumes_mean_centered_by_site.npy"),Xv)
np.save(os.path.join(OUTPUT_DATA,"features.npy"),features)


X = np.hstack([cov, Xv])


np.save(os.path.join(OUTPUT_DATA,"Xrois_volumes_mean_centered_by_site+cov.npy"),X)
np.save(os.path.join(OUTPUT_DATA,"y.npy"),y)

featuresAndCov = (["age",'sex_num','Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent',
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

np.save(os.path.join(OUTPUT_DATA,"featuresAndCov.npy"),featuresAndCov)


#######################################################################################################
#######################################################################################################
#######################################################################################################


INPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/population.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs"

INPUT_SCZCO_RH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_VIP_RH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/ROIs_analysis/freesurfer_stats/aparc_thickness_rh_all.csv"

INPUT_SCZCO_LH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/ROIs_analysis/freesurfer_stats/aparc_thickness_lh_all.csv"
INPUT_VIP_LH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/results/ROIs_analysis/freesurfer_stats/aparc_thickness_lh_all.csv"

#Create thickness dataset
#######################################################################################################
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
rh_thk_sczco = pd.read_csv(INPUT_SCZCO_RH_THICKNESS,sep='\t')
rh_thk_vip = pd.read_csv(INPUT_VIP_RH_THICKNESS,sep='\t')
rh_thk_all = rh_thk_sczco.append(rh_thk_vip, ignore_index=True)
rh_thk_all["id"] = rh_thk_all["rh.aparc.thickness"]

lh_thk_sczco = pd.read_csv(INPUT_SCZCO_LH_THICKNESS,sep='\t')
lh_thk_vip = pd.read_csv(INPUT_VIP_LH_THICKNESS,sep='\t')
lh_thk_all = lh_thk_sczco.append(lh_thk_vip, ignore_index=True)
lh_thk_all["id"] = lh_thk_all["lh.aparc.thickness"]

thk_all = rh_thk_all.merge(lh_thk_all,on="id")

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
for p in thk_all["id"]:
    print(os.path.basename(p))
    thk_all["id"][i] = os.path.basename(p)
    i=i+1

table = pop.merge(thk_all, on="id")
del table["lh.aparc.thickness"]
y = np.asarray(table["dx_num"])
cov = table[["age",'sex_num','site_num']]
cov = np.asarray(cov)
stats = np.asarray(table)
Xt = stats[:,13:].astype(float)
features = table.keys()[13:]
features =features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/Xrois_thickness.npy",Xt)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/features_thickness.npy",features)

X = np.hstack([cov, Xt])
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/Xrois_thick+cov.npy",X)

featuresAndCov = (["age",'sex_num','site_num','rh_bankssts_thickness', 'rh_caudalanteriorcingulate_thickness',
       'rh_caudalmiddlefrontal_thickness', 'rh_cuneus_thickness',
       'rh_entorhinal_thickness', 'rh_fusiform_thickness',
       'rh_inferiorparietal_thickness', 'rh_inferiortemporal_thickness',
       'rh_isthmuscingulate_thickness', 'rh_lateraloccipital_thickness',
       'rh_lateralorbitofrontal_thickness', 'rh_lingual_thickness',
       'rh_medialorbitofrontal_thickness', 'rh_middletemporal_thickness',
       'rh_parahippocampal_thickness', 'rh_paracentral_thickness',
       'rh_parsopercularis_thickness', 'rh_parsorbitalis_thickness',
       'rh_parstriangularis_thickness', 'rh_pericalcarine_thickness',
       'rh_postcentral_thickness', 'rh_posteriorcingulate_thickness',
       'rh_precentral_thickness', 'rh_precuneus_thickness',
       'rh_rostralanteriorcingulate_thickness',
       'rh_rostralmiddlefrontal_thickness', 'rh_superiorfrontal_thickness',
       'rh_superiorparietal_thickness', 'rh_superiortemporal_thickness',
       'rh_supramarginal_thickness', 'rh_frontalpole_thickness',
       'rh_temporalpole_thickness', 'rh_transversetemporal_thickness',
       'rh_insula_thickness', 'rh_MeanThickness_thickness', 'lh_bankssts_thickness',
       'lh_caudalanteriorcingulate_thickness',
       'lh_caudalmiddlefrontal_thickness', 'lh_cuneus_thickness',
       'lh_entorhinal_thickness', 'lh_fusiform_thickness',
       'lh_inferiorparietal_thickness', 'lh_inferiortemporal_thickness',
       'lh_isthmuscingulate_thickness', 'lh_lateraloccipital_thickness',
       'lh_lateralorbitofrontal_thickness', 'lh_lingual_thickness',
       'lh_medialorbitofrontal_thickness', 'lh_middletemporal_thickness',
       'lh_parahippocampal_thickness', 'lh_paracentral_thickness',
       'lh_parsopercularis_thickness', 'lh_parsorbitalis_thickness',
       'lh_parstriangularis_thickness', 'lh_pericalcarine_thickness',
       'lh_postcentral_thickness', 'lh_posteriorcingulate_thickness',
       'lh_precentral_thickness', 'lh_precuneus_thickness',
       'lh_rostralanteriorcingulate_thickness',
       'lh_rostralmiddlefrontal_thickness', 'lh_superiorfrontal_thickness',
       'lh_superiorparietal_thickness', 'lh_superiortemporal_thickness',
       'lh_supramarginal_thickness', 'lh_frontalpole_thickness',
       'lh_temporalpole_thickness', 'lh_transversetemporal_thickness',
       'lh_insula_thickness', 'lh_MeanThickness_thickness'])

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/data_ROIs/featureThksAndCov.npy",featuresAndCov)

