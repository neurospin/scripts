#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:19:57 2018

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
from nibabel import gifti
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from sklearn.mixture import GMM


##############################################################################
pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/\
VBM/population.csv")

site = np.load('/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy')




U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site_U0_100comp/\
U_scores_corrected/U_all.npy")

U_all = U_all[site==3]
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/y.npy")
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]

U_all_scz = scipy.stats.zscore(U_all_scz)


k_range = [1,13,19,20,22,62,89]

#mod = KMeans(n_clusters=3)
#mod.fit(U_all_scz[:,k_range])
#labels_all_scz = mod.labels_

mod = GMM(n_components=3)
labels_all_scz = mod.fit_predict(U_all_scz[:,k_range])

df = pd.DataFrame()
df["labels"] = labels_all_scz
df["age"] = pop_all["age"].values[y_all==1]
df["sex"] = pop_all["sex_num"].values[y_all==1]

for i in (k_range):
    df["U%s"%i] = U_all_scz[:,i]

LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)




#############################################################################
#PLOT WEIGHTS OF PC FOR EACH CLUSTER
df_complete = pd.DataFrame()
df_complete["PC"] = 99
df_complete["score"] = 99

ind = 0
for i in (df.index.values):
    for k in (k_range):
        df_complete = df_complete.append(df[df.index==i][['labels_name', 'age']],ignore_index=True)
        df_complete.loc[df_complete.index==ind,"PC"] ="PC %s"%k
        df_complete.loc[df_complete.index==ind,"score"] = df[df.index==i]['U%s'%k].values
        ind = ind +1

fig = plt.figure()
fig.set_size_inches(11.7, 8.27)
ax = sns.barplot(x="labels_name", y="score", hue="PC", data=df_complete,order=["cluster 1","cluster 2","cluster 3"])
plt.legend(loc ='lower left' )
#############################################################################


DATA_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = labels_all_scz

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
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values

        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                 df[df["labels"]==1][key],\
                 df[df["labels"]==2][key])
        if p<0.1:
            print(key)
            print(p)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = T
        df_stats.loc[df_stats.clinical_scores==key,"p"] = p

    except:
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan





key = "cvlsdfrr"
key = "cvlthits"
key = "d4prime"
key = "vocabsca"
key = "wcsttcom"
key = "fp1raw"
key = "ssblsraw"

df = pd.DataFrame()
score = df_scores[key].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df[key] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                     df[df["labels"]==1][key],\
                     df[df["labels"]==2][key])
ax = sns.violinplot(x="labels_name", y=key, data=df,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))




df_scores["sansTOTAL"] = df_scores["sans1"].astype(np.float).values+df_scores["sans2"].astype(np.float).values+\
df_scores["sans3"].astype(np.float).values+df_scores["sans4"].astype(np.float).values+\
df_scores["sans5"].astype(np.float).values+df_scores["sans6"].astype(np.float).values+\
df_scores["sans7"].astype(np.float).values+df_scores["sans8"].astype(np.float).values+\
df_scores["sans9"].astype(np.float).values+df_scores["sans10"].astype(np.float).values+\
df_scores["sans11"].astype(np.float).values+df_scores["sans12"].astype(np.float).values+\
df_scores["sans13"].astype(np.float).values+df_scores["sans14"].astype(np.float).values+\
df_scores["sans15"].astype(np.float).values+df_scores["sans16"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans18"].astype(np.float).values+\
df_scores["sans19"].astype(np.float).values+df_scores["sans20"].astype(np.float).values+\
df_scores["sans21"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans23"].astype(np.float).values+df_scores["sans24"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values

df_scores["sapsTOTAL"] = df_scores["sans1"].astype(np.float).values+df_scores["saps2"].astype(np.float).values+\
df_scores["saps3"].astype(np.float).values+df_scores["saps4"].astype(np.float).values+\
df_scores["saps5"].astype(np.float).values+df_scores["saps6"].astype(np.float).values+\
df_scores["saps7"].astype(np.float).values+df_scores["saps8"].astype(np.float).values+\
df_scores["saps9"].astype(np.float).values+df_scores["saps10"].astype(np.float).values+\
df_scores["saps11"].astype(np.float).values+df_scores["saps12"].astype(np.float).values+\
df_scores["saps13"].astype(np.float).values+df_scores["saps14"].astype(np.float).values+\
df_scores["saps15"].astype(np.float).values+df_scores["saps16"].astype(np.float).values+\
df_scores["saps17"].astype(np.float).values+df_scores["saps18"].astype(np.float).values+\
df_scores["saps19"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps21"].astype(np.float).values+df_scores["saps22"].astype(np.float).values+\
df_scores["saps23"].astype(np.float).values+df_scores["saps24"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps26"].astype(np.float).values+\
df_scores["saps27"].astype(np.float).values+df_scores["saps28"].astype(np.float).values+\
df_scores["saps29"].astype(np.float).values+df_scores["saps30"].astype(np.float).values+\
df_scores["saps31"].astype(np.float).values+df_scores["saps32"].astype(np.float).values+\
df_scores["saps33"].astype(np.float).values+df_scores["saps34"].astype(np.float).values

df_scores["sansSUbtot"] = df_scores["sans8"].astype(np.float).values+df_scores["sans13"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values

df_scores["sapsSUbtot"] = df_scores["saps7"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps34"].astype(np.float).values

df_scores["labels"]=labels_cluster
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df_scores["labels_name"]  = df_scores["labels"].map(LABELS_DICT)



################################################################################


df = pd.DataFrame()
score = df_scores["sansTOTAL"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["sansTOTAL"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["sansTOTAL"],\
                     df[df["labels"]==1]["sansTOTAL"],\
                     df[df["labels"]==2]["sansTOTAL"])
ax = sns.violinplot(x="labels_name", y="sansTOTAL", data=df,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))


df = pd.DataFrame()
score = df_scores["sapsTOTAL"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["sapsTOTAL"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["sapsTOTAL"],\
                     df[df["labels"]==1]["sapsTOTAL"],\
                     df[df["labels"]==2]["sapsTOTAL"])
ax = sns.violinplot(x="labels_name", y="sapsTOTAL", data=df,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))


df = pd.DataFrame()
score = df_scores["sansSUbtot"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["sansSUbtot"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["sansSUbtot"],\
                     df[df["labels"]==1]["sansSUbtot"],\
                     df[df["labels"]==2]["sansSUbtot"])
ax = sns.violinplot(x="labels_name", y="sansSUbtot", data=df,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))


df = pd.DataFrame()
score = df_scores["sapsSUbtot"].astype(np.float).values
df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["sapsSUbtot"] =  score[np.array(np.isnan(score)==False)]
T,p = scipy.stats.f_oneway(df[df["labels"]==0]["sapsSUbtot"],\
                     df[df["labels"]==1]["sapsSUbtot"],\
                     df[df["labels"]==2]["sapsSUbtot"])
ax = sns.violinplot(x="labels_name", y="sapsSUbtot", data=df,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
