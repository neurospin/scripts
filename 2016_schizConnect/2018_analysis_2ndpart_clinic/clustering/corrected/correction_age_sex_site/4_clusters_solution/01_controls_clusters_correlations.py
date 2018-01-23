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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"

site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")


pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop= pop[pop["site_num"]==3]
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/4_clusters_solution/with_controls/labels_all.npy")
labels_cluster = labels_cluster[site==3]


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
results/clustering/corrected_results/correction_age_sex_site/4_clusters_solution/\
with_controls/controls_clusters_clinics_p_values.csv"
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values

        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        T,p = scipy.stats.f_oneway(df[df["labels"]=="controls"][key],\
                                   df[df["labels"]==0][key],\
                 df[df["labels"]==1][key],\
                 df[df["labels"]==2][key],df[df["labels"]==3][key])
        print(p)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = T
        df_stats.loc[df_stats.clinical_scores==key,"p"] = p

    except:
        print("issue")
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(output)

################################################################################


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/4_clusters_solution/with_controls/nudast"

df = pd.DataFrame()
score = df_scores["vocabsca"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["vocabsca"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]=="controls"]["vocabsca"],\
                           df[df["labels"]==0]["vocabsca"],\
                     df[df["labels"]==1]["vocabsca"],\
                     df[df["labels"]==2]["vocabsca"],\
                    df[df["labels"]==3]["vocabsca"])
ax = sns.violinplot(x="labels_name", y="vocabsca", data=df,order=["controls","cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"vocabsca.png"))




df = pd.DataFrame()
score = df_scores["cvlfps"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["cvlfps"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]=="controls"]["cvlfps"],\
                            df[df["labels"]==0]["cvlfps"],\
                     df[df["labels"]==1]["cvlfps"],\
                     df[df["labels"]==2]["cvlfps"],\
df[df["labels"]==3]["cvlfps"])
ax = sns.violinplot(x="labels_name", y="cvlfps", data=df,order=["controls","cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"cvlfps.png"))


df = pd.DataFrame()
score = df_scores["matrxraw"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["matrxraw"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]=="controls"]["matrxraw"],\
                            df[df["labels"]==0]["matrxraw"],\
                     df[df["labels"]==1]["matrxraw"],\
                     df[df["labels"]==2]["matrxraw"],\
df[df["labels"]==3]["matrxraw"])
ax = sns.violinplot(x="labels_name", y="matrxraw", data=df,order=["controls","cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"matrxraw.png"))

df = pd.DataFrame()
score = df_scores["dstscalc"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["dstscalc"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]=="controls"]["dstscalc"],\
                            df[df["labels"]==0]["dstscalc"],\
                     df[df["labels"]==1]["dstscalc"],\
                     df[df["labels"]==2]["dstscalc"],\
df[df["labels"]==3]["dstscalc"])
ax = sns.violinplot(x="labels_name", y="dstscalc", data=df,order=["controls","cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"dstscalc.png"))

df = pd.DataFrame()
score = df_scores["sstscalc"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2", 2: "cluster 3", 3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["sstscalc"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]=="controls"]["sstscalc"],\
                            df[df["labels"]==0]["sstscalc"],\
                     df[df["labels"]==1]["sstscalc"],\
                     df[df["labels"]==2]["sstscalc"],\
df[df["labels"]==3]["sstscalc"])
ax = sns.violinplot(x="labels_name", y="sstscalc", data=df,order=["controls","cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"sstscalc.png"))

################################################################################

