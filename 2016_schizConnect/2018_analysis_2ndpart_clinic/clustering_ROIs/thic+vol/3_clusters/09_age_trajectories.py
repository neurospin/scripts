#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:08:00 2018

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

y_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/y.npy")
pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/population.csv")
age = pop["age"].values
age_scz = age[y_all==1]
age_con = age[y_all==0]

features_thickness = np.load("/neurospin/brainomics/2016_schizConnect\
/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/Xrois_thickness.npy")
features_volume = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/Xrois_volumes.npy")
features = np.concatenate((features_thickness,features_volume), axis=1)
features = features[:,(np.std(features, axis=0) > 1e-6)]

Thalamus_volume = features[:,74] + features[:,92]
Hippocampus_volume = features[:,81] + features[:,96]
Amygdala_volume = features[:,82] + features[:,97]
temporal_thickness = features[:,7]+features[:,13]+features[:,28]+features[:,42]+\
features[:,48]+features[:,62]
frontal_thickness = features[:,20]+features[:,22]+features[:,55]+features[:,57]

features_of_interest_name =  ['temporal_thickness',"frontal_thickness",\
               'hippocampus volume','amygdala volume','thalamus volume']
features = np.vstack((temporal_thickness,frontal_thickness,\
               Hippocampus_volume,Amygdala_volume,Thalamus_volume)).T


features = ((features - features[y_all==0,:].mean(axis=0))/features[y_all==0,:].std(axis=0))


######################################
#CORRECT FEATURES FOR AGE AND SITES FIRST
df = pd.DataFrame()
df["age"] = pop["age"].values
df["sex"] = pop["sex_num"].values
df["site"] = pop["site_num"].values

import statsmodels.api as sm
from statsmodels.formula.api import ols
i=0
sex_gender_corrected_features = features
features_of_interest_name =  ['temporal_thickness',"frontal_thickness",\
               'hippocampus_volume','amygdala_volume','thalamus_volume']
for f in features_of_interest_name:
    df[f] = features[:,i]
    mod = ols("%s ~ sex+C(site)"%f,data = df).fit()
    res = mod.resid
    sex_gender_corrected_features[:,i] = res
    print (mod.summary())
    i= i+1
features = sex_gender_corrected_features
######################################



features_scz = features[y_all==1,:]
features_con = features[y_all==0,:]





labels = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/thick+vol/3_clusters/labels_cluster.npy")
labels_scz = labels[y_all==1]

df_con = pd.DataFrame()
df_con["Age"] = age_con
df_con["Temporal thickness"] = features_con[:,0]
df_con["Frontal thickness"] = features_con[:,1]
df_con["Hippocampus volume"] = features_con[:,2]
df_con["Amygdala volume"] = features_con[:,3]
df_con["Thalamus volume"] = features_con[:,4]

df_scz_clust1 = pd.DataFrame()
df_scz_clust1["Age"] = age_scz[labels_scz == 'SCZ Cluster 1']
df_scz_clust1["Temporal thickness"] = features_scz[labels_scz == 'SCZ Cluster 1',0]
df_scz_clust1["Frontal thickness"] = features_scz[labels_scz == 'SCZ Cluster 1',1]
df_scz_clust1["Hippocampus volume"] = features_scz[labels_scz == 'SCZ Cluster 1',2]
df_scz_clust1["Amygdala volume"] = features_scz[labels_scz == 'SCZ Cluster 1',3]
df_scz_clust1["Thalamus volume"] = features_scz[labels_scz == 'SCZ Cluster 1',4]

df_scz_clust2 = pd.DataFrame()
df_scz_clust2["Age"] = age_scz[labels_scz == 'SCZ Cluster 2']
df_scz_clust2["Temporal thickness"] = features_scz[labels_scz == 'SCZ Cluster 2',0]
df_scz_clust2["Frontal thickness"] = features_scz[labels_scz == 'SCZ Cluster 2',1]
df_scz_clust2["Hippocampus volume"] = features_scz[labels_scz == 'SCZ Cluster 2',2]
df_scz_clust2["Amygdala volume"] = features_scz[labels_scz == 'SCZ Cluster 2',3]
df_scz_clust2["Thalamus volume"] = features_scz[labels_scz == 'SCZ Cluster 2',4]


df_scz_clust3 = pd.DataFrame()
df_scz_clust3["Age"] = age_scz[labels_scz == 'SCZ Cluster 3']
df_scz_clust3["Temporal thickness"] = features_scz[labels_scz == 'SCZ Cluster 3',0]
df_scz_clust3["Frontal thickness"] = features_scz[labels_scz == 'SCZ Cluster 3',1]
df_scz_clust3["Hippocampus volume"] = features_scz[labels_scz == 'SCZ Cluster 3',2]
df_scz_clust3["Amygdala volume"] = features_scz[labels_scz == 'SCZ Cluster 3',3]
df_scz_clust3["Thalamus volume"] = features_scz[labels_scz == 'SCZ Cluster 3',4]



output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/\
results/thick+vol/3_clusters/age_trajectories"
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/\
results/thick+vol/3_clusters/age_trajectories/corrected_age_gender"

import seaborn as sns

sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="Temporal thickness", data=df_con,label= "Controls",marker='o')
sns.regplot(x="Age", y="Temporal thickness", data=df_scz_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="Temporal thickness", data=df_scz_clust2,label= "SCZ Cluster 2",marker='o')
sns.regplot(x="Age", y="Temporal thickness", data=df_scz_clust3,label= "SCZ Cluster 3",marker='o')
plt.legend()
plt.savefig(os.path.join(output,"Temporal_thickness.png"))

sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="Frontal thickness", data=df_con,label= "Controls",marker='o')
sns.regplot(x="Age", y="Frontal thickness", data=df_scz_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="Frontal thickness", data=df_scz_clust2,label= "SCZ Cluster 2",marker='o')
sns.regplot(x="Age", y="Frontal thickness", data=df_scz_clust3,label= "SCZ Cluster 3",marker='o')
plt.legend()
plt.savefig(os.path.join(output,"Frontal_thickness.png"))


sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="Hippocampus volume", data=df_con,label= "Controls",marker='o')
sns.regplot(x="Age", y="Hippocampus volume", data=df_scz_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="Hippocampus volume", data=df_scz_clust2,label= "SCZ Cluster 2",marker='o')
sns.regplot(x="Age", y="Hippocampus volume", data=df_scz_clust3,label= "SCZ Cluster 3",marker='o')
plt.legend()
plt.savefig(os.path.join(output,"Hippocampus_volume.png"))


sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="Amygdala volume", data=df_con,label= "Controls",marker='o')
sns.regplot(x="Age", y="Amygdala volume", data=df_scz_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="Amygdala volume", data=df_scz_clust2,label= "SCZ Cluster 2",marker='o')
sns.regplot(x="Age", y="Amygdala volume", data=df_scz_clust3,label= "SCZ Cluster 3",marker='o')
plt.legend()
plt.savefig(os.path.join(output,"Amygdala_volume.png"))

sns.set(color_codes=True)
plt.figure()
sns.regplot(x="Age", y="Thalamus volume", data=df_con,label= "Controls",marker='o')
sns.regplot(x="Age", y="Thalamus volume", data=df_scz_clust1,label= "SCZ Cluster 1",marker='o')
sns.regplot(x="Age", y="Thalamus volume", data=df_scz_clust2,label= "SCZ Cluster 2",marker='o')
sns.regplot(x="Age", y="Thalamus volume", data=df_scz_clust3,label= "SCZ Cluster 3",marker='o')
plt.legend()
plt.savefig(os.path.join(output,"Thalamus_volume.png"))

