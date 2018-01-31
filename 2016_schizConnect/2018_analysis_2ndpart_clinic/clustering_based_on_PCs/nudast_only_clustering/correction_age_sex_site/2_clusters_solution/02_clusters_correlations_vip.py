#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:02:41 2018

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

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site_U0/2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster[site==4]

################################################################################



df_stats = pd.DataFrame(columns=["T","p"])
df_stats.insert(0,"clinical_scores",scores)################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_U0/2_clusters_solution/vip/\
clusters_clinics_p_values_vip.csv"
for key in scores:
    try:
        neurospycho = clinic[key].astype(np.float).values

        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                 df[df["labels"]==1][key])
        print(p)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = T
        df_stats.loc[df_stats.clinical_scores==key,"p"] = p

    except:
        print("issue")
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(output)

################################################################################
