#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:53:19 2017

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

DATA_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
clinic = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv")
clinic = clinic[clinic["dx"]==1]
pop = pd.read_csv(os.path.join(DATA_PATH,"pop_vip_scz.csv"))

age = pop["age"].values
sex = pop["sex_num"].values


scores = "CVLT_RC_listeA_tot","CVLT_P_listeA_tot","CVLT_I_listeA_tot",\
"CVLT_RICT_RC_tot","CVLT_RICT_P_tot","CVLT_RICT_I_tot",\
"WAIS_COMPL_IM_STD","WAIS_COMPL_IM_CR","WAIS_VOC_TOT","WAIS_VOC_STD",\
"WAIS_VOC_CR","WAIS_COD_tot","WAIS_COD_err","WAIS_COD_brut","WAIS_COD_CR",\
"WAIS_COD_STD","WAIS_SIMI_tot","WAIS_SIMI_STD","WAIS_SIMI_CR","NART33_Tot",\
"NART33_QIT","NART33_QIV","NART33_QIP","WAIS_CUB_TOT","WAIS_CUB_STD",\
"WAIS_CUB_CR","WAIS_ARITH_T0T","WAIS_ARITH_STD","WAIS_ARITH_CR","WAIS_MC_OD_TOT",\
"WAIS_MC_OINV_TOT","WAIS_MC_TOT","WAIS_MC_EMP_END","WAIS_MC_EMP_ENV","WAIS_MC_STD",\
"WAIS_MC_CR","WAIS_MC_EMP_END_STD","WAIS_MC_EMP_ENV_STD","WAIS_INFO_TOT","WAIS_INFO_STD",\
"WAIS_INFO_CR","WAIS_SLC_TOT","WAIS_SLC_STD","WAIS_SLC_CR","WAIS_ASS_OBJ_TOT","WAIS_ASS_OBJ_STD",\
"WAIS_ASS_OBJ_CR","WAIS_DET_MENT",


clusters = 'cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',\
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',\
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',\
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',\
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus'


df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",scores)
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/vip_clusters_clinics_p_values.csv"
for key in scores:
    try:
        neurospycho = clinic[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]
            del mod

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)
