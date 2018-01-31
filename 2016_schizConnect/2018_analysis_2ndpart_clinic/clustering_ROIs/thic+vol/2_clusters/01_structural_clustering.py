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
site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
##############################################################################

#1) Extarction of thickness features
##############################################################################

pop_thickness = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/thickness/corrected_results/data_corrected/pop_all_corrected.csv")
features_name_thickness = np.array(['rh_bankssts_thickness', 'rh_caudalanteriorcingulate_thickness',
       'rh_caudalmiddlefrontal_thickness', 'rh_cuneus_thickness',
       'rh_entorhinal_thickness', 'rh_fusiform_thickness',
       'rh_inferiorparietal_thickness', 'rh_inferiortemporal_thickness',
       'rh_isthmuscingulate_thickness', 'rh_lateraloccipital_thickness',
       'rh_lateralorbitofrontal_thickness', 'rh_lingual_thickness',
       'rh_medialorbitofrontal_thickness', 'rh_middletemporal_thickness',
       'rh_parahippocampal_thickness', 'rh_paracentral_thickness',
       'rh_parsopercularis_thickness', 'rh_parsorbitalis_thickness',
       'rh_parstriangularis_thickness', 'rh_pericalcarine_thickness',
       'rh_postcentral_thickness', 'rh_posteriorcingulate_thickness',
       'rh_precentral_thickness', 'rh_precuneus_thickness',
       'rh_rostralanteriorcingulate_thickness',
       'rh_rostralmiddlefrontal_thickness', 'rh_superiorfrontal_thickness',
       'rh_superiorparietal_thickness', 'rh_superiortemporal_thickness',
       'rh_supramarginal_thickness', 'rh_frontalpole_thickness',
       'rh_temporalpole_thickness', 'rh_transversetemporal_thickness',
       'rh_insula_thickness', 'rh_MeanThickness_thickness',
       'lh_bankssts_thickness', 'lh_caudalanteriorcingulate_thickness',
       'lh_caudalmiddlefrontal_thickness', 'lh_cuneus_thickness',
       'lh_entorhinal_thickness', 'lh_fusiform_thickness',
       'lh_inferiorparietal_thickness', 'lh_inferiortemporal_thickness',
       'lh_isthmuscingulate_thickness', 'lh_lateraloccipital_thickness',
       'lh_lateralorbitofrontal_thickness', 'lh_lingual_thickness',
       'lh_medialorbitofrontal_thickness', 'lh_middletemporal_thickness',
       'lh_parahippocampal_thickness', 'lh_paracentral_thickness',
       'lh_parsopercularis_thickness', 'lh_parsorbitalis_thickness',
       'lh_parstriangularis_thickness', 'lh_pericalcarine_thickness',
       'lh_postcentral_thickness', 'lh_posteriorcingulate_thickness',
       'lh_precentral_thickness', 'lh_precuneus_thickness',
       'lh_rostralanteriorcingulate_thickness',
       'lh_rostralmiddlefrontal_thickness', 'lh_superiorfrontal_thickness',
       'lh_superiorparietal_thickness', 'lh_superiortemporal_thickness',
       'lh_supramarginal_thickness', 'lh_frontalpole_thickness',
       'lh_temporalpole_thickness', 'lh_transversetemporal_thickness',
       'lh_insula_thickness', 'lh_MeanThickness_thickness'])

features_thickness = pop_thickness[features_name_thickness].as_matrix()
features_name_thickness = features_name_thickness[np.std(features_thickness, axis=0) > 1e-6]
##############################################################################

#1) Extarction of volume features
##############################################################################
pop_volume = pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/volume/corrected_results/data_corrected/pop_all_corrected.csv")
features_name_volume = np.array(['Left_Lateral_Ventricle', 'Left_Inf_Lat_Vent',
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

features_volume = pop_volume[features_name_volume].as_matrix()
features_name_volume = features_name_volume[np.std(features_volume, axis=0) > 1e-6]

##############################################################################

features = np.concatenate((features_thickness,features_volume), axis=1)
features_name = np.concatenate((features_name_thickness,features_name_volume))

features = features[:,(np.std(features, axis=0) > 1e-6)]
#sum(np.isnan(features)==True)

features = scipy.stats.zscore(features)

features_of_interest = [12,28,47,63,74,81,82,92,96,97]
features_of_interest_name = features_name[features_of_interest]
features = features[:,features_of_interest]


features_scz = features[y_all==1,:]
features_con = features[y_all==0,:]


mod = KMeans(n_clusters=2)
mod.fit(features_scz)


labels_all_scz = mod.labels_

df = pd.DataFrame()
df["age"] = pop_volume["age"].values
df["labels"] = np.nan
df["labels"][y_all==1] = labels_all_scz
df["labels"][y_all==0] = "controls"
LABELS_DICT = {"controls":"Controls",0: "SCZ Cluster 1", 1: "SCZ Cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)

i=0
for f in features_of_interest_name:
    df[f] = features[:,i]
    i= i+1


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/thick+vol/2_clusters"
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
ax = sns.barplot(x="labels_name", y="score", hue="Feature", data=df_complete,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.legend(loc ='lower left' )
plt.savefig(os.path.join(output,"cluster_weights.png"))



plt.figure()
sns.set_style("whitegrid")
ax = sns.barplot(x="labels_name", y="age",data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.savefig(os.path.join(output,"age.png"))

#############################################################################

#ANOVA on age

T, p = scipy.stats.f_oneway(df[df["labels"]==0]["age"],\
                     df[df["labels"]==1]["age"])
ax = sns.violinplot(x="labels_name", y="age", data=df,order=["Controls","SCZ Cluster 1","SCZ Cluster 2"])
plt.title("ANOVA patients: t = %s, and  p= %s"%(T,p))
plt.savefig(os.path.join(output,"age_anova.png"))