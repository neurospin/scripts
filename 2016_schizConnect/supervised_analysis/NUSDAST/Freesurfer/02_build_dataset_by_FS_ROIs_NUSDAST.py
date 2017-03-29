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

BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer"
INPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/population.csv"
INPUT_RH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/freesurfer_stats/aparc_thickness_rh_all.csv"
INPUT_LH_THICKNESS_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/freesurfer_stats/aparc_thickness_lh_all.csv"
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data"

#Create thickness dataset
pop = pd.read_csv(INPUT_CSV)
pop["lh.aparc.thickness"] = "/neurospin/abide/schizConnect/processed/freesurfer/NUSDAST_"+ pop.date + "_"+ pop.subjectid
rh_stats = pd.read_csv(INPUT_RH_THICKNESS_CSV,sep='\t')
rh_stats["lh.aparc.thickness"] = rh_stats["rh.aparc.thickness"]
lh_stats = pd.read_csv(INPUT_LH_THICKNESS_CSV,sep='\t')
stats = rh_stats.merge(lh_stats,on = "lh.aparc.thickness")
table = pop.merge(stats, on="lh.aparc.thickness")
stats = np.asarray(table)
Xth = stats[:,13:].astype(float)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/Xrois_thickness.npy",Xth)

      
# Create Volume dataset
pop = pd.read_csv(INPUT_CSV)
pop["Measure:volume"] = "/neurospin/abide/schizConnect/processed/freesurfer/NUSDAST_"+ pop.date + "_"+ pop.subjectid
volumes_stats = pd.read_csv(INPUT_VOLUME_CSV,sep='\t')
table = pop.merge(volumes_stats, on="Measure:volume")
stats = np.asarray(table)
Xv = stats[:,12:].astype(float)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/Xrois_volumes.npy",Xv)


# Create Volume + thicnkess dataset
Xtot = np.hstack([Xth,Xv])
np.save("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/data/Xrois_volumes+thickness.npy",Xv)


# #Check subjects that do not correpsond between images and clinic   
#for i, ID in enumerate(pop["lh.aparc.thickness"] ):
#    if ID not in list(stats["lh.aparc.thickness"]):
#        print (ID)
#        print ("is not in list")
