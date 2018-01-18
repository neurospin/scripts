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
U_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all.npy")
U_all_con = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_con.npy")
U_all_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/U_all_scz.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")

X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/data/X.npy")[:,2:]
X_all_mean_voxel_value = X_all.mean(axis=1)

df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["sex"] = pop_all["sex_num"].values
df["site"] = pop_all["site_num"].values
df["mean_brain"]=X_all_mean_voxel_value


for i in range(10):
    df["U%s"%i] = U_all[:,i]
    mod = ols("U%s ~ age+sex+C(site)+mean_brain"%i,data = df).fit()
    res = mod.resid
    df["U%s_corr"%i] = res
    print (mod.summary())

U_all_corr = df[['U0_corr','U1_corr','U2_corr','U3_corr','U4_corr',\
                    'U5_corr','U6_corr','U7_corr','U8_corr','U9_corr']].values
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_mean_VBM/\
U_scores_corrected/U_all.npy",U_all_corr)

for i in range(10):
    plt.figure()
    plt.plot(df["age"],df["U%s"%i],'o')
    plt.plot(df["age"],df["U%s_corr"%i],'o')
    plt.title("PC %s"%i)
#
for i in range(1,10):
    plt.figure()
    plt.plot(df["sex"],df["U%s"%i],'o')
    plt.plot(df["sex"],df["U%s_corr"%i],'o')


for i in range(1,10):
    plt.figure()
    plt.plot(df["site"],df["U%s"%i],'o')
    plt.plot(df["site"],df["U%s_corr"%i],'o')
 ###############################################################################
