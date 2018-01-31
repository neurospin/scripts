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
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_based_on_predictive_markers/data/pop_all.csv")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")

features_name = ['cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus']
features = pop_all[features_name].as_matrix()



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


features_corr = df[['age','sex','site','cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus']]

features_corr.to_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_based_on_predictive_markers/results/corrected_results/\
data_corrected/pop_all_corrected.csv",)

 ###############################################################################
