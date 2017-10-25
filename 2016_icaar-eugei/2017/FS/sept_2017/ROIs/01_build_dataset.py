#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:44:12 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import mulm
from mulm import MUOLS
import brainomics

BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei'
INPUT_ROI_CSV = '/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_stats/aseg_volume_all.csv'
TEMPLATE_PATH = os.path.join(BASE_PATH, "preproc_FS/freesurfer_template")

INPUT_CSV = os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","population.csv")
OUTPUT = os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR","data","data_ROIs")


pop = pd.read_csv(INPUT_CSV)
vol_roi = pd.read_csv(INPUT_ROI_CSV,sep='\t')
vol_roi["image"] = vol_roi['Measure:volume']

table = pop.merge(vol_roi,on = "image")
y = np.asarray(table["group_outcom.num"])

cov = table[["age",'sex.num']]
cov = cov.fillna(cov.mean())
cov = np.asarray(cov)

stats = np.asarray(table)
Xv = stats[:,14:].astype(float)
features = table.keys()[14:]
features = features.get_values()


X = np.hstack([cov, Xv])

X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (39, 68)
np.save("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/Freesurfer/ICAAR/data/data_ROIs/Xrois_vol+cov.npy",X)
np.save("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/Freesurfer/ICAAR/data/data_ROIs/y.npy",y)

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

np.save("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/Freesurfer/ICAAR/data/data_ROIs/featuresAndCov.npy",featuresAndCov)
#######################################################################################################
