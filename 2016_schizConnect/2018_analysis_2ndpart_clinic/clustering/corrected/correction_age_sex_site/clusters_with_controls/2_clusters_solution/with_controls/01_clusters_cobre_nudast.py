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

site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")


pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop= pop[pop["site_num"]==1]
age = pop["age"].values
sex = pop["sex_num"].values

labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/with_controls/labels_all.npy")
labels_cluster = labels_cluster[site==1]


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


key_of_interest = list()

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
        if p<0.05:
            print(key)
            print(p)
            key_of_interest.append(key)
    except:
        print("issue")
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/with_controls/cobre"

for key in key_of_interest:
    plt.figure()
    df = pd.DataFrame()
    score = df_scores[key].astype(np.float).values
    df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
    LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2"}
    df["labels_name"]  = df["labels"].map(LABELS_DICT)
    df[key] =  score[np.array(np.isnan(score)==False)]
    T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],
                         df[df["labels"]==1][key])
    ax = sns.violinplot(x="labels_name", y=key, data=df,order=["controls","cluster 1","cluster 2"])
    plt.title("ANOVA patients diff: t = %s, and  p= %s"%(T,p))
    plt.savefig(os.path.join(output,"%s.png"%key))


################################################################################
DICT = {"grade 6 or less":	1,
"grade 7 - 12 (without graduating high school)":	2,
"graduated high school or high school equivalent":	3,
"part college":	4,
"graduated 2 yr college":	5,
"graduated 4 yr college":	6,
"part graduate/professional school":	7,
"completed graduate/professional school":	8}

df_scores["CODEM_5"]  = df_scores["CODEM_5"].map(DICT)
df_scores["CODEM_6"]  = df_scores["CODEM_6"].map(DICT)