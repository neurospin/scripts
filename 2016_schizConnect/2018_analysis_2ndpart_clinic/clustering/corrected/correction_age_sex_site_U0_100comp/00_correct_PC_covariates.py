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
U_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/data_100comp/U_all.npy")
U_all_con = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/data_100comp/U_all_con.npy")
U_all_scz = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data/data_100comp/U_all_scz.npy")


pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")



df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["sex"] = pop_all["sex_num"].values
df["site"] = pop_all["site_num"].values
df["U0"] = U_all[:,0]


for i in range(100):
    df["U%s"%i] = U_all[:,i]
    mod = ols("U%s ~ age+sex+C(site)+U0"%i,data = df).fit()
    res = mod.resid
    df["U%s_corr"%i] = res
    print (mod.summary())

U_all_corr = df[['U0_corr','U1_corr','U2_corr','U3_corr','U4_corr','U5_corr','U6_corr','U7_corr','U8_corr','U9_corr',\
                 'U10_corr','U11_corr','U12_corr','U13_corr','U14_corr','U15_corr','U16_corr','U17_corr','U18_corr','U19_corr',\
                 'U20_corr','U21_corr','U22_corr','U23_corr','U24_corr','U25_corr','U26_corr','U27_corr','U28_corr','U29_corr',\
                 'U30_corr','U31_corr','U32_corr','U33_corr','U34_corr','U35_corr','U36_corr','U37_corr','U38_corr','U39_corr',\
                 'U40_corr','U41_corr','U42_corr','U43_corr','U44_corr','U45_corr','U46_corr','U47_corr','U48_corr','U49_corr',\
                 'U50_corr','U51_corr','U52_corr','U53_corr','U54_corr','U55_corr','U56_corr','U57_corr','U58_corr','U59_corr',\
                 'U60_corr','U61_corr','U62_corr','U63_corr','U64_corr','U65_corr','U66_corr','U67_corr','U68_corr','U69_corr',\
                 'U70_corr','U71_corr','U72_corr','U73_corr','U74_corr','U75_corr','U76_corr','U77_corr','U78_corr','U79_corr',\
                 'U80_corr','U81_corr','U82_corr','U83_corr','U84_corr','U85_corr','U86_corr','U87_corr','U88_corr','U89_corr',\
                 'U90_corr','U91_corr','U92_corr','U93_corr','U94_corr','U95_corr','U96_corr','U97_corr','U98_corr','U99_corr']].values

np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_U0_100comp/\
U_scores_corrected/U_all.npy",U_all_corr)

