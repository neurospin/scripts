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


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site_mean_VBM/4_clusters_solution/labels_cluster.npy")
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
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_mean_VBM/4_clusters_solution"


df = pd.DataFrame()
score = df_scores["vocabsca"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["vocabsca"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["vocabsca"],\
                     df[df["labels"]==1]["vocabsca"],\
                     df[df["labels"]==2]["vocabsca"],\
                     df[df["labels"]==3]["vocabsca"])
ax = sns.violinplot(x="labels", y="vocabsca", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"vocabsca.png"))

df = pd.DataFrame()
score = df_scores["d4prime"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["d4prime"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["d4prime"],\
                     df[df["labels"]==1]["d4prime"],\
                     df[df["labels"]==2]["d4prime"],\
                     df[df["labels"]==3]["d4prime"])
ax = sns.violinplot(x="labels", y="d4prime", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"d4prime.png"))


df = pd.DataFrame()
score = df_scores["lmiscalc"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["lmiscalc"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]==0]["lmiscalc"],\
                     df[df["labels"]==1]["lmiscalc"],\
                     df[df["labels"]==2]["lmiscalc"],\
                     df[df["labels"]==3]["lmiscalc"])
ax = sns.violinplot(x="labels", y="lmiscalc", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"lmiscalc.png"))

df = pd.DataFrame()
score = df_scores["matrxsca"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["matrxsca"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]==0]["matrxsca"],\
                     df[df["labels"]==1]["matrxsca"],\
                     df[df["labels"]==2]["matrxsca"],\
                     df[df["labels"]==3]["matrxsca"])
ax = sns.violinplot(x="labels", y="matrxsca", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"matrxsca.png"))

df = pd.DataFrame()
score = df_scores["wcstpsvr"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["wcstpsvr"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]==0]["wcstpsvr"],\
                     df[df["labels"]==1]["wcstpsvr"],\
                     df[df["labels"]==2]["wcstpsvr"],\
                     df[df["labels"]==3]["wcstpsvr"])
ax = sns.violinplot(x="labels", y="wcstpsvr", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"wcstpsvr.png"))

df = pd.DataFrame()
score = df_scores["lnsscalc"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["lnsscalc"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]==0]["lnsscalc"],\
                     df[df["labels"]==1]["lnsscalc"],\
                     df[df["labels"]==2]["lnsscalc"],\
                     df[df["labels"]==3]["lnsscalc"])
ax = sns.violinplot(x="labels", y="lnsscalc", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"lnsscalc.png"))

df = pd.DataFrame()
score = df_scores["cvlfps"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
df["cvlfps"] =  score[np.array(np.isnan(score)==False)]
T, p = scipy.stats.f_oneway(df[df["labels"]==0]["cvlfps"],\
                     df[df["labels"]==1]["cvlfps"],\
                     df[df["labels"]==2]["cvlfps"],\
                     df[df["labels"]==3]["cvlfps"])
ax = sns.violinplot(x="labels", y="cvlfps", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"cvlfps.png"))

################################################################################
