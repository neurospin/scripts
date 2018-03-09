import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from nibabel import gifti
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


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

##############################################################################
Medialorbitofrontal_thickness = features[:,12] + features[:,47]
Superiortemporal_thickness = features[:,28] + features[:,63]
Frontalpole_thickness = features[:,65] + features[:,30]
Temporalpole_thickness = features[:,66] + features[:,31]
Thalamus_volume = features[:,74] + features[:,92]
Hippocampus_volume = features[:,81] + features[:,96]
Amygdala_volume = features[:,82] + features[:,97]
ICV = features[:,-1]
Mean_thickness = features[:,34]+features[:,66]
posteriorcingulate_thickness = features[:,21]+features[:,56]

temporal_thickness = features[:,7]+features[:,13]+features[:,28]+features[:,42]+\
features[:,48]+features[:,62]
frontal_thickness = features[:,20]+features[:,22]+features[:,55]+features[:,57]
parietal_thickness = features[:,6]+features[:,27]+features[:,41]+features[:,61]
precuneus_thickness = features[:,23]+features[:,58]


features_of_interest_name =  ['temporal_thickness',"frontal_thickness",\
               'hippocampus volume','amygdala volume','thalamus volume']
features = np.vstack((temporal_thickness,frontal_thickness,\
               Hippocampus_volume,Amygdala_volume,Thalamus_volume)).T

features = ((features - features[y_all==0,:].mean(axis=0))/features[y_all==0,:].std(axis=0))
features_scz = features[y_all==1,:]
features_con = features[y_all==0,:]

#clustering
##############################################################################
mod = KMeans(n_clusters=3)
mod.fit(features_scz)
labels_all_scz_orig = mod.labels_
centers_orig =mod.cluster_centers_



features_cluster0 = list()
features_cluster1 = list()
features_cluster2 = list()
n_ite = 1000
for ite in range(n_ite ):
    mod = KMeans(n_clusters=3)
    index = np.arange(sum(y_all==1))
    index_resampled = np.random.choice(index, size=253, replace=True)
    mod.fit(features_scz[index_resampled,:])
    labels_all_scz = mod.labels_
    centers = mod.cluster_centers_
    for k_orig in range(3):
        distance = list()
        for i in range(3):
            distance.append(np.linalg.norm(centers[i,:]-centers_orig[k_orig,:]))
        argmin = np.argmin(distance)
        f = features_scz[index_resampled,:][labels_all_scz==argmin]
        if k_orig==0:features_cluster0.append(f)
        if k_orig==1:features_cluster1.append(f)
        if k_orig==2:features_cluster2.append(f)

assert len(features_cluster0) ==len(features_cluster1) ==len(features_cluster2) == n_ite



##############################################################################
T_scz1 = np.zeros((5,n_ite))
T_scz2 = np.zeros((5,n_ite))
T_scz3 = np.zeros((5,n_ite))
p_scz1 = np.zeros((5,n_ite))
p_scz2 = np.zeros((5,n_ite))
p_scz3 = np.zeros((5,n_ite))
for ite in range(n_ite ):
    con = features_con
    scz1 = features_cluster0[ite]
    scz2 = features_cluster1[ite]
    scz3 = features_cluster2[ite]
    T_scz1[:,ite],p_scz1[:,ite] = scipy.stats.ttest_ind(con,scz1)
    T_scz2[:,ite],p_scz2[:,ite] = scipy.stats.ttest_ind(con,scz2)
    T_scz3[:,ite],p_scz3[:,ite] = scipy.stats.ttest_ind(con,scz3)


df_scz1 = pd.DataFrame()
df_scz1 ["T-statistics"] = np.hstack((T_scz1[0],T_scz1[1],T_scz1[2],T_scz1[3],T_scz1[4]))
df_scz1 ["ROIs"] = np.hstack((np.repeat("Temporal\n Thickness", n_ite),np.repeat("Frontal\n Thickness", n_ite),\
np.repeat("Hippocampus\n Volume", n_ite),np.repeat("Amygdala\n Volume", n_ite),np.repeat("Thalamus\n Volume", n_ite)))
df_scz1["Clusters"] = np.repeat("Cluster 1",5*n_ite)

df_scz2 = pd.DataFrame()
df_scz2["T-statistics"] = np.hstack((T_scz2[0],T_scz2[1],T_scz2[2],T_scz2[3],T_scz2[4]))
df_scz2["ROIs"] = np.hstack((np.repeat("Temporal\n Thickness", n_ite),np.repeat("Frontal\n Thickness", n_ite),\
np.repeat("Hippocampus\n Volume", n_ite),np.repeat("Amygdala\n Volume", n_ite),np.repeat("Thalamus\n Volume", n_ite)))
df_scz2["Clusters"] = np.repeat("Cluster 2",5*n_ite)

df_scz3 = pd.DataFrame()
df_scz3["T-statistics"] = np.hstack((T_scz3[0],T_scz3[1],T_scz3[2],T_scz3[3],T_scz3[4]))
df_scz3["ROIs"] = np.hstack((np.repeat("Temporal\n Thickness", n_ite),np.repeat("Frontal\n Thickness", n_ite),\
np.repeat("Hippocampus\n Volume", n_ite),np.repeat("Amygdala\n Volume", n_ite),np.repeat("Thalamus\n Volume", n_ite)))
df_scz3["Clusters"] = np.repeat("Cluster 3",5*n_ite)

df = pd.concat((df_scz1,df_scz2,df_scz3))
sns.set_style("whitegrid")
ax = sns.factorplot(x="ROIs", y="T-statistics", data=df, kind="bar",col="Clusters")
plt.savefig("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/thick+vol/\
3_clusters/boostrap_T_stat.png")

