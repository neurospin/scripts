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
site_scz = site[y_all==1]
site_con = site[y_all==0]


U_all = scipy.stats.zscore(U_all)
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]

#Cluster only SCZ and check position of controls
#############################################################################
#
#mod = KMeans(n_clusters=2)
#mod.fit(U_all_scz[:,])
#labels_all = mod.labels_
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/2_clusters_solution"
#np.save(os.path.join(output,"labels_cluster.npy"),labels_all)

labels_all = np.load(os.path.join(output,"labels_cluster.npy"))


#############################################################################

df = pd.DataFrame()
df["labels"] = labels_all
df["age"] = pop_all["age"].values[y_all==1]
df["site"] = pop_all["site_num"].values[y_all==1]
df["sex"] = pop_all["sex_num"].values[y_all==1]
df["U0"] = U_all[:,0][y_all==1]

df_con = pd.DataFrame()
df_con["age"] = pop_all["age"].values[y_all==0]
df_con["U0"] = U_all[:,0][y_all==0]



#############################################################################
#ALL
sns.distplot(df_con["U0"],label="Controls")
sns.distplot(df["U0"][labels_all==0],label="SCZ Cluster 1")
sns.distplot(df["U0"][labels_all==1],label="SCZ Cluster 2")
plt.legend()
plt.savefig(os.path.join(output,"clusters_dist.png"))
#############################################################################
#ALL

sum(labels_all==0) #134 in cluster 1
sum(labels_all==1) #142 in cluster 2

df["age"][labels_all==0].mean()
df["age"][labels_all==1].mean()

df["age"][labels_all==0].std()
df["age"][labels_all==1].std()

sum(df["sex"][labels_all==0]==0)
sum(df["sex"][labels_all==0]==1)

sum(df["sex"][labels_all==1]==0)
sum(df["sex"][labels_all==1]==1)


sum(df["site"][labels_all==0]==1)
sum(df["site"][labels_all==0]==2)
sum(df["site"][labels_all==0]==3)
sum(df["site"][labels_all==0]==4)


sum(df["site"][labels_all==1]==1)
sum(df["site"][labels_all==1]==2)
sum(df["site"][labels_all==1]==3)
sum(df["site"][labels_all==1]==4)

#############################################################################

#COBRE
labels_all_cobre  =labels_all[site_scz==1]


sum(labels_all[site_scz==1]==0)
sum(labels_all[site_scz==1]==1)

df["age"][site_scz==1][labels_all_cobre==0].mean()
df["age"][site_scz==1][labels_all_cobre==1].mean()

df["age"][site_scz==1][labels_all_cobre==0].std()
df["age"][site_scz==1][labels_all_cobre==1].std()

sum(df["sex"][site_scz==1][labels_all_cobre==0]==0)
sum(df["sex"][site_scz==1][labels_all_cobre==0]==1)

sum(df["sex"][site_scz==1][labels_all_cobre==1]==0)
sum(df["sex"][site_scz==1][labels_all_cobre==1]==1)




output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre"


sns.distplot(df_con["U0"][site_scz==1],label="Controls")
sns.distplot(df["U0"][site_scz==1],label="SCZ")
plt.legend()


sns.distplot(df_con["U0"][site_con==3],label="COBRE Controls")
sns.distplot(df["U0"][site_scz==1][labels_all_cobre==0],label="COBRE SCZ Cluster 1")
sns.distplot(df["U0"][site_scz==1][labels_all_cobre==1],label="COBRE SCZ Cluster 2")
plt.legend()
plt.savefig(os.path.join(output,"clusters_dist.png"))

#############################################################################
#NUDAST
labels_all_nudast  =labels_all[site_scz==3]


sum(labels_all[site_scz==3]==0)
sum(labels_all[site_scz==3]==1)

df["age"][site_scz==3][labels_all_nudast==0].mean()
df["age"][site_scz==3][labels_all_nudast==1].mean()

df["age"][site_scz==3][labels_all_nudast==0].std()
df["age"][site_scz==3][labels_all_nudast==1].std()

sum(df["sex"][site_scz==3][labels_all_nudast==0]==0)
sum(df["sex"][site_scz==3][labels_all_nudast==0]==1)

sum(df["sex"][site_scz==3][labels_all_nudast==1]==0)
sum(df["sex"][site_scz==3][labels_all_nudast==1]==1)


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nudast"


sns.distplot(df_con["U0"][site_scz==3],label="Controls")
sns.distplot(df["U0"][site_scz==3],label="SCZ")
plt.legend()


sns.distplot(df_con["U0"][site_con==3],label="NUSDAST Controls")
sns.distplot(df["U0"][site_scz==3][labels_all_nudast==0],label="NUSDAST SCZ Cluster 1")
sns.distplot(df["U0"][site_scz==3][labels_all_nudast==1],label="NUSDAST SCZ Cluster 2")
plt.legend()
plt.savefig(os.path.join(output,"clusters_dist.png"))

#############################################################################
#NMORPH
labels_all_nmorph  =labels_all[site_scz==2]


sum(labels_all[site_scz==2]==0)
sum(labels_all[site_scz==2]==1)

df["age"][site_scz==2][labels_all_nmorph==0].mean()
df["age"][site_scz==2][labels_all_nmorph==1].mean()

df["age"][site_scz==2][labels_all_nmorph==0].std()
df["age"][site_scz==2][labels_all_nmorph==1].std()

sum(df["sex"][site_scz==2][labels_all_nmorph==0]==0)
sum(df["sex"][site_scz==2][labels_all_nmorph==0]==1)

sum(df["sex"][site_scz==2][labels_all_nmorph==1]==0)
sum(df["sex"][site_scz==2][labels_all_nmorph==1]==1)

output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nmorph"


sns.distplot(df_con["U0"][site_scz==2],label="Controls")
sns.distplot(df["U0"][site_scz==2],label="SCZ")
plt.legend()


sns.distplot(df_con["U0"][site_con==2],label="NMORPH Controls")
sns.distplot(df["U0"][site_scz==2][labels_all_nmorph==0],label="NMORPH SCZ Cluster 1")
sns.distplot(df["U0"][site_scz==2][labels_all_nmorph==1],label="NMORPH SCZ Cluster 2")
plt.legend()
plt.savefig(os.path.join(output,"clusters_dist.png"))



#############################################################################
#VIP
labels_all_nmorph  =labels_all[site_scz==2]


sum(labels_all[site_scz==2]==0)
sum(labels_all[site_scz==2]==1)

df["age"][site_scz==4][labels_all_vip==0].mean()
df["age"][site_scz==4][labels_all_vip==1].mean()

df["age"][site_scz==4][labels_all_vip==0].std()
df["age"][site_scz==4][labels_all_vip==1].std()

sum(df["sex"][site_scz==4][labels_all_vip==0]==0)
sum(df["sex"][site_scz==4][labels_all_vip==0]==1)

sum(df["sex"][site_scz==4][labels_all_vip==1]==0)
sum(df["sex"][site_scz==4][labels_all_vip==1]==1)

output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/vip"


sns.distplot(df_con["U0"][site_scz==4],label="Controls")
sns.distplot(df["U0"][site_scz==4],label="SCZ")
plt.legend()


sns.distplot(df_con["U0"][site_con==4],label="VIP Controls")
sns.distplot(df["U0"][site_scz==4][labels_all_vip==0],label="VIP SCZ Cluster 1")
sns.distplot(df["U0"][site_scz==4][labels_all_vip==1],label="VIP SCZ Cluster 2")
plt.legend()
plt.savefig(os.path.join(output,"clusters_dist.png"))



#############################################################################
##nudast+nmorph
#sum(labels_all[site_scz!=4]==0) #19in cluster 1
#sum(labels_all[site_scz[site_scz!=4]!=4]==1) #20 in cluster 2
#
#labels_all_nudast+nmorph  =labels_all[site_scz!=4]
#labels_all_nudast+nmorph  =labels_all[site_scz[site_scz!=4]!=1]
#
#output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
#results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
#2_clusters_solution/nudast+nmorph"
#
#
#sns.distplot(df_con["U0"][site_scz==4],label="Controls")
#sns.distplot(df["U0"][site_scz==4],label="SCZ")
#plt.legend()
#
#
#sns.distplot(df_con["U0"][site_con==4],label="nudast+nmorph Controls")
#sns.distplot(df["U0"][site_scz==4][labels_all_nudast+nmorph==0],label="nudast+nmorph SCZ Cluster 1")
#sns.distplot(df["U0"][site_scz==4][labels_all_nudast+nmorph==1],label="nudast+nmorph SCZ Cluster 2")
#plt.legend()
#plt.savefig(os.path.join(output,"clusters_dist.png"))
