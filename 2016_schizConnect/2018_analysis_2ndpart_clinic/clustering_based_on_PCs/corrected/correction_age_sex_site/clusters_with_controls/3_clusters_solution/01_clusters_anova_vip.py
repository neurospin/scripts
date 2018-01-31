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


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_vip_scz.csv"))
age = pop["age"].values
sex = pop["sex_num"].values


y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster[site==4]




##Demogrpahic symptoms
#################################################################################
#
#sum(sex[labels_cluster==0]==0)
#sum(sex[labels_cluster==0]==1)
#
#sum(sex[labels_cluster==1]==0)
#sum(sex[labels_cluster==1]==1)
#
#age[labels_cluster==0].mean()
#age[labels_cluster==1].mean()
#age[labels_cluster==0].std()
#age[labels_cluster==1].std()
#scipy.stats.f_oneway(age[labels_cluster==0],age[labels_cluster==1])
##################################################################################

################################################################################
################################################################################

df_stats = pd.DataFrame(columns=["T","p"])
df_stats.insert(0,"clinical_scores",clinic.keys())
################################################################################
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/vip/clusters_clinics_p_values.csv"

for key in clinic.keys():
    try:
        neurospycho = clinic[key].astype(np.float).values

        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                 df[df["labels"]==1][key])
        if p<0.05:
            print(key)
            print(p)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = T
        df_stats.loc[df_stats.clinical_scores==key,"p"] = p

    except:
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(output)
################################################################################

################################################################################



################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/vip/anova"



###############################################################################
key = "PANSS_GALPSYCHOPAT"
key = "PANSS_POSITIVE"
key = "PANSS_NEGATIVE"

key = "AXE1_AAO"

plt.figure()
df = pd.DataFrame()
score = clinic[key].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df[key] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                     df[df["labels"]==1][key])
ax = sns.violinplot(x="labels_name", y=key, data=df,order=["cluster 1","cluster 2"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"%s.png"%key))


###############################################################################
