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

output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/2_clusters_solution"
#np.save(os.path.join(output,"labels_cluster.npy"),labels_all)

labels_all = np.load(os.path.join(output,"labels_cluster.npy"))

#Save labels_controls
#############################################################################
df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["sex"] = pop_all["sex_num"].values
df["site"] = pop_all["site_num"].values
df["labels"] = np.nan
df["labels"][y_all==1] = labels_all
df["labels"][y_all==0] = "controls"


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/with_controls"

#np.save(os.path.join(output,"labels_all.npy"),df["labels"].values)

labels_with_controls = df["labels"].values

#############################################################################
#ALL

df = pd.DataFrame()
df["labels"] = labels_with_controls
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["U0"] = U_all[:,0]


sns.violinplot(x="labels_name",y="U0",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend()
plt.savefig(os.path.join(output,"clusters_violin_plot.png"))

#############################################################################


#COBRE

labels_with_controls_cobre  =labels_with_controls[site==1]



df = pd.DataFrame()
df["labels"] = labels_with_controls[site==1]
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["U0"] = U_all[:,0][site==1]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre"

sns.violinplot(x="labels_name",y="U0",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend()
plt.savefig(os.path.join(output,"clusters_violin_plot.png"))


#############################################################################
#NUDAST
labels_with_controls_nudast  =labels_with_controls[site==3]



df = pd.DataFrame()
df["labels"] = labels_with_controls[site==3]
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["U0"] = U_all[:,0][site==3]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nudast"

sns.violinplot(x="labels_name",y="U0",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend()
plt.savefig(os.path.join(output,"clusters_violin_plot.png"))

#############################################################################
#NMORPH
labels_with_controls_nmorph  =labels_with_controls[site==2]

df = pd.DataFrame()
df["labels"] = labels_with_controls[site==2]
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["U0"] = U_all[:,0][site==2]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nmorph"

sns.violinplot(x="labels_name",y="U0",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend()
plt.savefig(os.path.join(output,"clusters_violin_plot.png"))




#############################################################################
#VIP
labels_with_controls_vip  =labels_with_controls[site==4]

df = pd.DataFrame()
df["labels"] = labels_with_controls[site==4]
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)
df["U0"] = U_all[:,0][site==4]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/vip"

sns.violinplot(x="labels_name",y="U0",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend()
plt.savefig(os.path.join(output,"clusters_violin_plot.png"))



