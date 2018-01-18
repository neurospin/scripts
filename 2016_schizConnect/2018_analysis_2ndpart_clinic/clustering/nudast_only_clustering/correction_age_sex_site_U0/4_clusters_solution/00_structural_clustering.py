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
U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/nudast_only_clustering/correction_age_sex_site_U0/U_scores_corrected.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/y.npy")
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/\
VBM/population.csv")

U_all_scz = (U_all_scz  -U_all_scz.mean())  / U_all_scz.std()

mod = KMeans(n_clusters=4)
mod.fit(U_all_scz[:,:])
labels_all_scz = mod.labels_

df = pd.DataFrame()
df["labels"] = labels_all_scz
df["age"] = pop_all["age"].values[y_all==1]
df["sex"] = pop_all["sex_num"].values[y_all==1]

for i in range(10):
    df["U%s"%i] = U_all_scz[:,i]

LABELS_DICT = {0: "cluster 1", 1: "cluster 2", 2: "cluster 3",3: "cluster 4"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/nudast_only_clustering/correction_age_sex_site_U0/4_clusters_solution"
np.save(os.path.join(output,"labels_cluster.npy"),labels_all_scz)


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
ax = sns.barplot(x="labels_name", y="score", hue="PC", data=df_complete,order=["cluster 1","cluster 2","cluster 3","cluster 4"])
plt.legend(loc ='lower left' )
plt.savefig(os.path.join(output,"cluster_weights.png"))

plt.figure()
sns.set_style("whitegrid")
ax = sns.barplot(x="labels_name", y="age",data=df,order=["cluster 1","cluster 2","cluster 3","cluster 4"])
plt.savefig(os.path.join(output,"age.png"))

#############################################################################

#ANOVA on age

T, p = scipy.stats.f_oneway(df[df["labels"]==0]["age"],\
                     df[df["labels"]==1]["age"],\
                     df[df["labels"]==2]["age"],\
                     df[df["labels"]==3]["age"])
ax = sns.violinplot(x="labels_name", y="age", data=df,order=["cluster 1","cluster 2","cluster 3","cluster 4"])
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"age_anova.png"))


