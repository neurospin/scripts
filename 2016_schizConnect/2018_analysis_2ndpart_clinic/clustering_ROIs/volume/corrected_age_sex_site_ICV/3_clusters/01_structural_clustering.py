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
y_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/y.npy")

pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/corrected_results/data_corrected/pop_all_corrected.csv")

site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
features_name = np.array(['Left_Lateral_Ventricle', 'Left_Inf_Lat_Vent',
       'Left_Cerebellum_White_Matter', 'Left_Cerebellum_Cortex',
       'Left_Thalamus_Proper', 'Left_Caudate', 'Left_Putamen',
       'Left_Pallidum',"third_Ventricle",'fourth_Ventricle', 'Brain_Stem',
       'Left_Hippocampus', 'Left_Amygdala', 'CSF', 'Left_Accumbens_area',
       'Left_VentralDC', 'Left_vessel', 'Left_choroid_plexus',
       'Right_Lateral_Ventricle', 'Right_Inf_Lat_Vent',
       'Right_Cerebellum_White_Matter', 'Right_Cerebellum_Cortex',
       'Right_Thalamus_Proper', 'Right_Caudate', 'Right_Putamen',
       'Right_Pallidum', 'Right_Hippocampus', 'Right_Amygdala',
       'Right_Accumbens_area', 'Right_VentralDC', 'Right_vessel',
       'Right_choroid_plexus', 'fifth_Ventricle', 'WM_hypointensities',
       'Left_WM_hypointensities', 'Right_WM_hypointensities',
       'non_WM_hypointensities', 'Left_non_WM_hypointensities',
       'Right_non_WM_hypointensities', 'Optic_Chiasm', 'CC_Posterior',
       'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior',
       'BrainSegVol', 'BrainSegVolNotVent', 'BrainSegVolNotVentSurf',
       'lhCortexVol', 'rhCortexVol', 'CortexVol',
       'lhCorticalWhiteMatterVol', 'rhCorticalWhiteMatterVol',
       'CorticalWhiteMatterVol', 'SubCortGrayVol', 'TotalGrayVol',
       'SupraTentorialVol', 'SupraTentorialVolNotVent',
       'SupraTentorialVolNotVentVox', 'MaskVol', 'BrainSegVol_to_eTIV',
       'MaskVol_to_eTIV', 'lhSurfaceHoles', 'rhSurfaceHoles',
       'SurfaceHoles', 'EstimatedTotalIntraCranialVol'])

features = pop_all[features_name].as_matrix()
features_name = features_name[np.std(features, axis=0) > 1e-6]

features = features[:,(np.std(features, axis=0) > 1e-6)]
#sum(np.isnan(features)==True)

features = scipy.stats.zscore(features)

features_of_interest = [4,11,12,22,26,27]
features_of_interest_name = features_name[features_of_interest]
features = features[:,features_of_interest]


features_scz = features[y_all==1,:]
features_con = features[y_all==0,:]


mod = KMeans(n_clusters=3)
mod.fit(features_scz[:,:])

labels_all_scz = mod.labels_

df = pd.DataFrame()
df["age"] = pop_all["age"].values
df["labels"] = np.nan
df["labels"][y_all==1] = labels_all_scz
df["labels"][y_all==0] = "controls"
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2", 2: "SCZ Cluster 3"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)

i=0
for f in features_of_interest_name:
    df[f] = features[:,i]
    i= i+1

output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/corrected_results/3_clusters"
np.save(os.path.join(output,"labels_cluster.npy"),df["labels_name"].values)

#############################################################################
#PLOT WEIGHTS OF PC FOR EACH CLUSTER
df_complete = pd.DataFrame()
df_complete["Feature"] = 99
df_complete["score"] = 99

ind = 0
for i in (df.index.values):
    for k in features_of_interest_name :
        df_complete = df_complete.append(df[df.index==i][['labels_name', 'age']],ignore_index=True)
        df_complete.loc[df_complete.index==ind,"Feature"] = k
        df_complete.loc[df_complete.index==ind,"score"] = df[df.index==i][k].values
        ind = ind +1

fig = plt.figure()
fig.set_size_inches(11.7, 8.27)
ax = sns.barplot(x="labels_name", y="score", hue="Feature", data=df_complete,order=["Controls","SCZ Cluster 1","SCZ Cluster 2","SCZ Cluster 3"])
plt.legend(loc ='lower left' )
plt.savefig(os.path.join(output,"cluster_weights.png"))



plt.figure()
sns.set_style("whitegrid")
ax = sns.barplot(x="labels_name", y="age",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2","SCZ Cluster 3"])
plt.savefig(os.path.join(output,"age.png"))

#############################################################################

#ANOVA on age

T, p = scipy.stats.f_oneway(df[df["labels"]==0]["age"],\
                     df[df["labels"]==1]["age"],\
 df[df["labels"]==2]["age"])
ax = sns.violinplot(x="labels_name", y="age", data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2","SCZ Cluster 3"])
plt.title("ANOVA patients: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"age_anova.png"))