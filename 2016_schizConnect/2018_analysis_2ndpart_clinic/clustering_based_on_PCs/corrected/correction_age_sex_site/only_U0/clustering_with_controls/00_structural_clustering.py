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


##############################################################################
U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")



U_all = scipy.stats.zscore(U_all)
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]


#Cluster with controls
#############################################################################

mod = KMeans(n_clusters=2)
mod.fit(U_all[:,])
labels_all = mod.labels_
df = pd.DataFrame()
df["labels"] = labels_all
df["age"] = pop_all["age"].values
df["U0"] = U_all[:,0]
df["dx"] = y_all



output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/only_U0/clustering_with_controls"
np.save(os.path.join(output,"labels_cluster.npy"),labels_all)




sum(labels_all[y_all==1]==0)
sum(labels_all[y_all==0]==0)
#Cluster 1: 198 controls and 96 SCZ
sum(labels_all[y_all==1]==1)
sum(labels_all[y_all==0]==1)
#Cluster 2: 132 controls and 180 SCZ


#Cluster 1; 67% of controls
#Cluster 2: 57% of SCZ

#35% of SCZ are in cluster 1, 65% of SCZ are in cluster 2
#60% of SCZ are in cluster 1, 40% of SCZ are in cluster 2


plt.plot(df["U0"][y_all==1][labels_all[y_all==1]==0],'o',label = "SCZ")
plt.plot(df["U0"][y_all==1][labels_all[y_all==1]==1],'o',label = "SCZ")
plt.plot(df["U0"][y_all==0][labels_all[y_all==0]==0],'o',label = "Controls")
plt.plot(df["U0"][y_all==0][labels_all[y_all==0]==1],'o',label = "Controls")
plt.legend()


plt.plot(labels_all,df["U0"],'o')

sns.violinplot(x = "labels",y="U0",data = df)
plt.plot
sns.distplot(df["U0"][y_all==0],label="CONTROLS")
sns.distplot(df["U0"][y_all==1],label="SCZ")
plt.legend()


sns.distplot(df["U0"][labels_all==0],label="Cluster 1")
sns.distplot(df["U0"][labels_all==1],label="Cluster 2")
plt.legend()

#############################################################################
###########################################################################

#ANOVA on age

T, p = scipy.stats.f_oneway(df[df["labels"]==0]["age"],\
                     df[df["labels"]==1]["age"])
ax = sns.violinplot(x="labels", y="age", data=df)
plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"age_anova.png"))