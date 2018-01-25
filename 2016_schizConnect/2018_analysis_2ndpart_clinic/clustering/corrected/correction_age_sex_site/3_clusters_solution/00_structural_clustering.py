#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:08:33 2017

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

##############################################################################
U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")

U_all = scipy.stats.zscore(U_all)


U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]



mod = KMeans(n_clusters=3)
#mod = AgglomerativeClustering(n_clusters=3)
mod.fit(U_all_scz[:,2:])
labels_all_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = labels_all_scz
df["age"] = pop_all["age"].values[y_all==1]
df["sex"] = pop_all["sex_num"].values[y_all==1]
df["site"] = pop_all["site_num"].values[y_all==1]

for i in range(10):
    df["U%s"%i] = U_all_scz[:,i]

LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)

#
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/3_clusters_solution"
#np.save(os.path.join(output,"labels_cluster.npy"),labels_all_scz)

labels_all_scz = np.load(os.path.join(output,"labels_cluster.npy"))

#############################################################################
#PLOT WEIGHTS OF PC FOR EACH CLUSTER
df_complete = pd.DataFrame()
df_complete["PC"] = 99
df_complete["score"] = 99

ind = 0
for i in (df.index.values):
    for k in range(10):
        df_complete = df_complete.append(df[df.index==i][['labels_name', 'age']],ignore_index=True)
        df_complete.loc[df_complete.index==ind,"PC"] ="PC %s"%k
        df_complete.loc[df_complete.index==ind,"score"] = df[df.index==i]['U%s'%k].values
        ind = ind +1

fig = plt.figure()
fig.set_size_inches(11.7, 8.27)
ax = sns.barplot(x="labels_name", y="score", hue="PC", data=df_complete,order=["cluster 1","cluster 2","cluster 3"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.legend(loc ='lower left' )
plt.savefig(os.path.join(output,"cluster_weights.png"))

plt.figure()
sns.set_style("whitegrid")
ax = sns.barplot(x="labels_name", y="age",data=df)
plt.savefig(os.path.join(output,"age.png"))

#############################################################################

#ANOVA on age

T, p = scipy.stats.f_oneway(df[df["labels"]==0]["age"],\
                     df[df["labels"]==1]["age"],\
                     df[df["labels"]==2]["age"])
ax = sns.violinplot(x="labels", y="age", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"age_anova.png"))

#Chi square test on sex

scipy.stats.chisquare(sum(df[df["labels"]==0]["sex"]),sum(df[df["labels"]==1]["sex"]),sum(df[df["labels"]==2]["sex"]))

#############################################################################

df.groupby("labels_name").mean()
df.groupby("labels_name").std()
df.groupby("labels_name").count()

df[df["labels"]==0]["age"].mean()
df[df["labels"]==1]["age"].mean()
df[df["labels"]==2]["age"].mean()

df[df["labels"]==0]["age"].std()
df[df["labels"]==1]["age"].std()
df[df["labels"]==2]["age"].std()


sum(df[df["labels"]==0]["sex"]==0)
sum(df[df["labels"]==0]["sex"]==1)
sum(df[df["labels"]==1]["sex"]==0)
sum(df[df["labels"]==1]["sex"]==1)
sum(df[df["labels"]==2]["sex"]==0)
sum(df[df["labels"]==2]["sex"]==1)


sum(df[df["labels"]==0]["site"]==1)
sum(df[df["labels"]==0]["site"]==2)
sum(df[df["labels"]==0]["site"]==3)
sum(df[df["labels"]==0]["site"]==4)

sum(df[df["labels"]==1]["site"]==1)
sum(df[df["labels"]==1]["site"]==2)
sum(df[df["labels"]==1]["site"]==3)
sum(df[df["labels"]==1]["site"]==4)

sum(df[df["labels"]==2]["site"]==1)
sum(df[df["labels"]==2]["site"]==2)
sum(df[df["labels"]==2]["site"]==3)
sum(df[df["labels"]==2]["site"]==4)

#############################################################################
for i in range(10):
    print(i+1)
    print(df[df["labels"]==0]["U%s"%i].mean())
    print(df[df["labels"]==1]["U%s"%i].mean())
    print(df[df["labels"]==2]["U%s"%i].mean())
    print("ANOVA results")
    T,p = scipy.stats.f_oneway(df[df["labels"]==0]["U%s"%i],\
                     df[df["labels"]==1]["U%s"%i],\
                     df[df["labels"]==2]["U%s"%i])
    print(T)
    print(p)


