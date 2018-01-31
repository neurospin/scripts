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
y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/data_by_site/NUSDAST/y.npy")
U = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/data/U_nudast.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/\
VBM/population.csv")

X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/data_by_site/NUSDAST/X.npy")[:,2:]

df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["sex"] = pop_all["sex_num"].values


for i in range(10):
    df["U%s"%i] = U[:,i]
    mod = ols("U%s ~ age+sex"%i,data = df).fit()
    res = mod.resid
    df["U%s_corr"%i] = res
    print (mod.summary())

U_corr = df[['U0_corr','U1_corr','U2_corr','U3_corr','U4_corr',\
                    'U5_corr','U6_corr','U7_corr','U8_corr','U9_corr']].values
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/nudast_only_clustering/correction_age_sex_site/\
U_scores_corrected",U_corr)

