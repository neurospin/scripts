"""
Created on Tue Oct 18 14:25:41 2016

@author: ad247405

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import pandas as pd
import nibabel
import brainomics.image_atlas
import mulm
import nilearn
from nilearn import plotting
from mulm import MUOLS



INPUT_VBM_X = "/neurospin/brainomics/2016_icaar-eugei/september_2017/VBM/ICAAR/data/X.npy"
INPUT_CAARMS_ALL_X = "/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/data/X_CAARMS_all.npy"
INPUT_CAARMS_FREQ_X = "/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/data/X_CAARMS_frequence.npy"
INPUT_CAARMS_SEVE_X = "/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/data/X_CAARMS_severity.npy"


X_vbm = np.load(INPUT_VBM_X)
X_caarms = np.load(INPUT_CAARMS_ALL_X)
X_freq = np.load(INPUT_CAARMS_FREQ_X)
X_seve = np.load(INPUT_CAARMS_SEVE_X)



X_vbm_caarms =  np.hstack((X_caarms,X_vbm))
X_vbm_caarms_freq =  np.hstack((X_freq,X_vbm))
X_vbm_caarms_seve =  np.hstack((X_seve,X_vbm))


np.save("/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS+VBM/data/X_vbm+caarms.npy",X_vbm_caarms)
np.save("/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS+VBM/data/X_vbm+caarms_freq.npy",X_vbm_caarms_freq)
np.save("/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS+VBM/data/X_vbm+caarms_seve.noy",X_vbm_caarms_seve)