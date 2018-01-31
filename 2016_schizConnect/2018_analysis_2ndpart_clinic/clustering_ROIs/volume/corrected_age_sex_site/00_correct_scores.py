#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 14:53:31 2018

@author: Amicie
"""

import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from nibabel import gifti
from sklearn.cluster import KMeans
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Remove effect of age and sex for all datasets

###############################################################################
y_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/y.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/population.csv")
site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")


features = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/Xrois_volumes.npy")
features_name = ['Left_Lateral_Ventricle', 'Left_Inf_Lat_Vent',
       'Left_Cerebellum_White_Matter', 'Left_Cerebellum_Cortex',
       'Left_Thalamus_Proper', 'Left_Caudate', 'Left_Putamen',
       'Left_Pallidum',"third_Ventricle",'fourth_Ventricle', 'Brain_Stem',
       'Left_Hippocampus', 'Left_Amygdala', 'CSF', 'Left_Accumbens_area',
       'Left_VentralDC', 'Left_vessel', 'Left_choroid_plexus',
       'Right_Lateral_Ventricle', 'Right_Inf_Lat_Vent',
       'Right_Cerebellum_White_Matter', 'Right_Cerebellum_Cortex',
       'Right_Thalamus_Proper', 'Right_Caudate', 'Right_Putamen',
       'Right_Pallidum', 'Right_Hippocampus', 'Right_Amygdala',
       'Right_Accumbens_area', 'Right_VentralDC', 'Right_vessel',
       'Right_choroid_plexus', 'fifth_Ventricle', 'WM_hypointensities',
       'Left_WM_hypointensities', 'Right_WM_hypointensities',
       'non_WM_hypointensities', 'Left_non_WM_hypointensities',
       'Right_non_WM_hypointensities', 'Optic_Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf',
       'lhCortexVol', 'rhCortexVol', 'CortexVol',
       'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
       'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
       'SupraTentorialVol', 'SupraTentorialVolNotVent',
       'SupraTentorialVolNotVentVox', 'MaskVol', 'BrainSegVol_to_eTIV',
       'MaskVol_to_eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles',
       'SurfaceHoles', 'EstimatedTotalIntraCranialVol']

df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["sex"] = pop_all["sex_num"].values
df["site"] = pop_all["site_num"].values

i=0
for f in features_name:
    df[f] = features[:,i]
    mod = ols("%s ~ age+sex+C(site)"%f,data = df).fit()
    res = mod.resid
    df["%s"%f] = res
    print (mod.summary())
    i= i+1


features_corr = df[['age','sex','site','Left_Lateral_Ventricle', 'Left_Inf_Lat_Vent',
       'Left_Cerebellum_White_Matter', 'Left_Cerebellum_Cortex',
       'Left_Thalamus_Proper', 'Left_Caudate', 'Left_Putamen',
       'Left_Pallidum',"third_Ventricle",'fourth_Ventricle', 'Brain_Stem',
       'Left_Hippocampus', 'Left_Amygdala', 'CSF', 'Left_Accumbens_area',
       'Left_VentralDC', 'Left_vessel', 'Left_choroid_plexus',
       'Right_Lateral_Ventricle', 'Right_Inf_Lat_Vent',
       'Right_Cerebellum_White_Matter', 'Right_Cerebellum_Cortex',
       'Right_Thalamus_Proper', 'Right_Caudate', 'Right_Putamen',
       'Right_Pallidum', 'Right_Hippocampus', 'Right_Amygdala',
       'Right_Accumbens_area', 'Right_VentralDC', 'Right_vessel',
       'Right_choroid_plexus', 'fifth_Ventricle', 'WM_hypointensities',
       'Left_WM_hypointensities', 'Right_WM_hypointensities',
       'non_WM_hypointensities', 'Left_non_WM_hypointensities',
       'Right_non_WM_hypointensities', 'Optic_Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf',
       'lhCortexVol', 'rhCortexVol', 'CortexVol',
       'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
       'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
       'SupraTentorialVol', 'SupraTentorialVolNotVent',
       'SupraTentorialVolNotVentVox', 'MaskVol', 'BrainSegVol_to_eTIV',
       'MaskVol_to_eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles',
       'SurfaceHoles', 'EstimatedTotalIntraCranialVol']]

features_corr.to_csv("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/\
corrected_results/data_corrected/pop_all_corrected.csv")

 ###############################################################################
