#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:25:32 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns


DATA_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data/data_panss"



cobre_vip_panss = pd.read_csv(os.path.join(DATA_PATH,"cobre+vip_panss.csv"))
nudast_nmorph_panss = pd.read_csv(os.path.join(DATA_PATH,"nudast+nmorph_panss.csv"))

#CONVERSIOn of SAPS AND SANS INTO PANSS POS AND PANS NEG
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3966195/
nudast_nmorph_panss["panss_pos"] = 11.1886 + (0.2587 * nudast_nmorph_panss["saps_tot"])
nudast_nmorph_panss["panss_neg"] = 7.1196 + (0.3362 *  nudast_nmorph_panss["sans_tot"])

nudast_nmorph_panss["sum_pos"]  = nudast_nmorph_panss["panss_pos"]
nudast_nmorph_panss["sum_neg"]  = nudast_nmorph_panss["panss_neg"]

panss_all = cobre_vip_panss.append(nudast_nmorph_panss)

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data/pop_all_scz.csv")
age = pop_all['age']
sex = pop_all['sex_num']
site = pop_all['site_num']


clusters = 'cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',\
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',\
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',\
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',\
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus'


df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",["sum_pos","sum_neg"])
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/all_clusters_clinics_p_values.csv"
for key in ("sum_pos","sum_neg"):
    try:
        neurospycho = panss_all[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df["site"] = site[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop_all[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex+ C(site)"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)
