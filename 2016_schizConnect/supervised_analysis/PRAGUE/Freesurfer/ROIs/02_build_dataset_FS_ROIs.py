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

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/data/FS/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = os.path.join(BASE_PATH,"data","data_ROIs")


# Create Volume dataset
pop = pd.read_csv(INPUT_CSV)
volumes_stats = pd.read_csv(INPUT_VOLUME_CSV,sep='\t')
for i in range(volumes_stats.shape[0]):
    volumes_stats.loc[volumes_stats.index==i,"code"] = "ESO"+volumes_stats["Measure:volume"][i][-11:-5]


table = pop.merge(volumes_stats, on="code")
assert table.shape == (132, 79)

y = np.asarray(table['dx_num'])
cov = table[["age",'sex_01code']]
cov = np.asarray(cov)
stats = np.asarray(table)
assert table.shape == (132, 79)

ROIs =  (["age",'sex_01code',
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
assert table.shape == (132, 49)
features = table.keys()
stats = np.asarray(table)
X = stats.astype(float)
assert X.shape == (132,49)

#Center by site
X = X - X.mean(axis=0)
assert X.shape == (132,49)

np.save(os.path.join(OUTPUT_DATA,"Xrois_volumes_mean_centered_by_site+cov.npy"),X)
np.save(os.path.join(OUTPUT_DATA,"y.npy"),y)
np.save(os.path.join(OUTPUT_DATA,"features.npy"),features)


###############################################################################
###############################################################################
###############################################################################
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
OUTPUT_DATA = os.path.join(BASE_PATH,"data","data_ROIs")

INPUT_SCZCO_RH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/\
data/FS/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_SCZCO_LH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/\
data/FS/freesurfer_stats/aparc_thickness_lh_all.csv"
INPUT_SCZCO_VOLUME ="/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/\
data/FS/freesurfer_stats/aseg_volume_all.csv"
#Create thickness dataset
#######################################################################################################
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
rh_thk = pd.read_csv(INPUT_SCZCO_RH_THICKNESS,sep='\t')
rh_thk["code"] = rh_thk["rh.aparc.thickness"]

lh_thk = pd.read_csv(INPUT_SCZCO_LH_THICKNESS,sep='\t')
lh_thk["code"] = lh_thk["lh.aparc.thickness"]

thk_all = rh_thk.merge(lh_thk,on="code")

i=0
for p in pop["mri_path_lh"]:
    print(os.path.basename(p)[:-7])
    pop["code"][i] = os.path.basename(p)[:-7]
    i=i+1
i=0
for p in thk_all["code"]:
    print(os.path.basename(p))
    thk_all["code"][i] = os.path.basename(p)
    i=i+1

table = pop.merge(thk_all, on="code")
del table["lh.aparc.thickness"]
y = np.asarray(table["dx_num"])
stats = np.asarray(table)
Xt = stats[:,13:].astype(float)
features = table.keys()[13:]
features =features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer/data/data_ROIs/Xrois_thickness.npy",Xt)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer/data/data_ROIs/features_thickness.npy",features)



#Create volume dataset
#######################################################################################################
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
vol = pd.read_csv(INPUT_SCZCO_VOLUME,sep='\t')


#################################################
i=0
for p in pop["mri_path_lh"]:
    print(os.path.basename(p)[:-7])
    pop["code"][i] = os.path.basename(p)[:-7]
    i=i+1
####################################################
i=0
vol["code"] = vol["Measure:volume"]
for p in vol["Measure:volume"]:
    print(os.path.basename(p))
    vol["code"][i] = os.path.basename(p)
    i=i+1


table = pop.merge(vol, on="code")
y = np.asarray(table["dx_num"])
stats = np.asarray(table)
Xv = stats[:,13:].astype(float)
features = table.keys()[13:]
features =features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer/data/data_ROIs/Xrois_volumes.npy",Xv)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/Freesurfer/data/data_ROIs/features_volumes.npy",features)



