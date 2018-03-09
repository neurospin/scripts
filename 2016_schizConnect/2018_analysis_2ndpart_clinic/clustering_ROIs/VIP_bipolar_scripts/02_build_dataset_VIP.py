#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:59:22 2017

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


BASE_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar"
INPUT_CSV = os.path.join(BASE_PATH,"pop_bp_hc.csv")
OUTPUT_DATA = os.path.join(BASE_PATH,"pop_bp_hc.csv")

INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/\
results/ROIs_analysis/freesurfer_stats/aseg_volume_all.csv"
INPUT_RH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/\
results/ROIs_analysis/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_LH_THICKNESS = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/\
results/ROIs_analysis/freesurfer_stats/aparc_thickness_lh_all.csv"
#Create thickness dataset
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
#np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar/y_vip_bp.npy",pop["dx_num"].values)

rh_thk = pd.read_csv(INPUT_RH_THICKNESS,sep='\t')
rh_thk["nip"] = rh_thk["rh.aparc.thickness"]

lh_thk = pd.read_csv(INPUT_LH_THICKNESS,sep='\t')
lh_thk["nip"] = lh_thk["lh.aparc.thickness"]

thk_all = rh_thk.merge(lh_thk,on="nip")

i=0
for p in pop["mri_path_lh"]:
    print(os.path.basename(p)[:-7])
    pop["nip"][i] = os.path.basename(p)[:-7]
    i=i+1
i=0
for p in thk_all["nip"]:
    print(os.path.basename(p))
    thk_all["nip"][i] = os.path.basename(p)
    i=i+1

table = pop.merge(thk_all, on="nip")
del table["lh.aparc.thickness"]
y = np.asarray(table["dx_num"])
stats = np.asarray(table)
Xt = stats[:,8:].astype(float)
features = table.keys()[8:]
features = features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar/Xrois_thickness.npy",Xt)
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar/features_thickness.npy",features)



#Create volume dataset
#######################################################################################################
#######################################################################################################
#######################################################################################################
pop = pd.read_csv(INPUT_CSV)
vol = pd.read_csv(INPUT_VOLUME_CSV,sep='\t')


#################################################
i=0
for p in pop["mri_path_lh"]:
    print(os.path.basename(p)[:-7])
    pop["nip"][i] = os.path.basename(p)[:-7]
    i=i+1
####################################################
i=0
vol["nip"] = vol["Measure:volume"]
for p in vol["Measure:volume"]:
    print(os.path.basename(p))
    vol["nip"][i] = os.path.basename(p)
    i=i+1


table = pop.merge(vol, on="nip")
y = np.asarray(table["dx_num"])
stats = np.asarray(table)
Xv = stats[:,8:].astype(float)
features = table.keys()[8:]
features =features.get_values()
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar/Xrois_volumes.npy",Xv)
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/vip_bipolar/features_volumes.npy",features)
