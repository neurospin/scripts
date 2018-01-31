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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_COBRE_assessmentData_4495.csv"

pop = pd.read_csv(os.path.join(DATA_PATH,"pop_cobre_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site_U0/2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster[site==3]


PANSS_MAP = {"Absent": 1, "Minimal": 2, "Mild": 3, "Moderate": 4, "Moderate severe": 5, "Severe": 6, "Extreme": 7,\
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
clinic["question_value"] = clinic["question_value"].map(PANSS_MAP)

df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]

################################################################################



df_stats = pd.DataFrame(columns=["T","p"])
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_U0/2_clusters_solution/cobre/\
clusters_clinics_p_values_cobre.csv"
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values

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

df_scores["PANSS_POS"] = df_scores["FIPAN_1"].astype(np.float).values+df_scores["FIPAN_2"].astype(np.float).values+\
df_scores["FIPAN_3"].astype(np.float).values+df_scores["FIPAN_4"].astype(np.float).values+\
df_scores["FIPAN_5"].astype(np.float).values+df_scores["FIPAN_6"].astype(np.float).values+\
df_scores["FIPAN_7"].astype(np.float).values


df_scores["PANSS_NEG"] = df_scores["FIPAN_8"].astype(np.float).values+df_scores["FIPAN_9"].astype(np.float).values+\
df_scores["FIPAN_10"].astype(np.float).values+df_scores["FIPAN_11"].astype(np.float).values+\
df_scores["FIPAN_12"].astype(np.float).values+df_scores["FIPAN_13"].astype(np.float).values+\
df_scores["FIPAN_14"].astype(np.float).values

df_scores["PANSS_DES"] = df_scores["FIPAN_15"].astype(np.float).values+df_scores["FIPAN_16"].astype(np.float).values+\
df_scores["FIPAN_17"].astype(np.float).values+df_scores["FIPAN_18"].astype(np.float).values+\
df_scores["FIPAN_19"].astype(np.float).values+df_scores["FIPAN_20"].astype(np.float).values+\
df_scores["FIPAN_21"].astype(np.float).values+df_scores["FIPAN_22"].astype(np.float).values+\
df_scores["FIPAN_23"].astype(np.float).values+df_scores["FIPAN_24"].astype(np.float).values+\
df_scores["FIPAN_25"].astype(np.float).values+df_scores["FIPAN_26"].astype(np.float).values+\
df_scores["FIPAN_27"].astype(np.float).values+df_scores["FIPAN_28"].astype(np.float).values+\
df_scores["FIPAN_29"].astype(np.float).values+df_scores["FIPAN_20"].astype(np.float).values

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_U0/2_clusters_solution/cobre/"

df = pd.DataFrame()
score = df_scores["PANSS_POS"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["PANSS_POS"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["PANSS_POS"],\
                     df[df["labels"]==1]["PANSS_POS"])
ax = sns.violinplot(x="labels", y="PANSS_POS", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"PANSS_POS.png"))


df = pd.DataFrame()
score = df_scores["PANSS_NEG"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["PANSS_NEG"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["PANSS_NEG"],\
                     df[df["labels"]==1]["PANSS_NEG"])
ax = sns.violinplot(x="labels", y="PANSS_NEG", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"PANSS_NEG.png"))

df = pd.DataFrame()
score = df_scores["PANSS_DES"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["PANSS_DES"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["PANSS_DES"],\
                     df[df["labels"]==1]["PANSS_DES"])
ax = sns.violinplot(x="labels", y="PANSS_DES", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"PANSS_DES.png"))


################################################################################
