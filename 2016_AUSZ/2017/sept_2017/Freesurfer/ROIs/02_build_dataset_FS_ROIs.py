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

BASE_PATH = "/neurospin/brainomics/2016_AUSZ/september_2017/results/Freesurfer"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
INPUT_VOLUME_CSV = "/neurospin/brainomics/2016_AUSZ/data/ROIs/freesurfer_stats/aseg_volume_all.csv"
OUTPUT_DATA = os.path.join(BASE_PATH,"ROIs_analysis","data")

#LP130187

# Create Volume dataset
pop = pd.read_csv(INPUT_CSV)
volumes_stats = pd.read_csv(INPUT_VOLUME_CSV,sep='\t')
for i in range(volumes_stats.shape[0]):
    volumes_stats.loc[volumes_stats.index==i,"IRM.1"] = volumes_stats["Measure:volume"][i][:-3]

volumes_stats.loc[volumes_stats["IRM.1"] == "Lp130187","IRM.1"] = "LP130187"

table = pop.merge(volumes_stats, on="IRM.1")
assert table.shape == (123, 77)

y = np.asarray(table['group.num'])
MASCtot = np.asarray(table[' MASCtot'])
DX = np.asarray(table['group.num'])

cov = table[["Âge",'sex.num']]
cov = np.asarray(cov)
stats = np.asarray(table)
assert table.shape == (123, 77)

ROIs =  (["Âge",'sex.num',
       'Left-Cerebellum-White-Matter', 'Left-Cerebellum-Cortex',
       'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
       'Left-Pallidum','Left-Hippocampus', 'Left-Amygdala', 'CSF', 'Left-Accumbens-area',
       'Left-VentralDC', 'Left-choroid-plexus',
       'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
       'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen',
       'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
       'Right-Accumbens-area', 'Right-VentralDC',
       'Right-choroid-plexus', 'Optic-Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'CortexVol','TotalGrayVol', 'EstimatedTotalIntraCranialVol'])

table = table[ROIs]
assert table.shape == (123, 49)
features = table.keys()
stats = np.asarray(table)
X = stats.astype(float)
assert X.shape == (123, 49)


np.save(os.path.join(OUTPUT_DATA,"X.npy"),X)
np.save(os.path.join(OUTPUT_DATA,"y.npy"),y)
np.save(os.path.join(OUTPUT_DATA,"DX.npy"),DX)
np.save(os.path.join(OUTPUT_DATA,"MASCtot.npy"),MASCtot)
np.save(os.path.join(OUTPUT_DATA,"features.npy"),features)



np.save(os.path.join(OUTPUT_DATA,"X_patients.npy"),X[DX!=0,:])
np.save(os.path.join(OUTPUT_DATA,"y_patients.npy"),y[DX!=0])
np.save(os.path.join(OUTPUT_DATA,"DX_patients.npy"),DX[DX!=0])
np.save(os.path.join(OUTPUT_DATA,"MASCtot_patients.npy"),MASCtot[DX!=0])


