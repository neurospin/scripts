#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:42:06 2017

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

BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all"
MASK_PATH = "/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy"
COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"


COMP_PATH = os.path.join(BASE_PATH,"results_3triscotte","0","struct_pca_0.01_0.5_0.1","components.npz")

components = np.load(COMP_PATH)["arr_0"]
assert components.shape == (299879, 10)


################################################################################
################################################################################
#Project hcp subjects
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/hcp/population.csv"
pop_hcp  = pd.read_csv(INPUT_CSV)

X_hcp = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/hcp/X.npy")

mask_mesh = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/mask.npy")


#Analysis cluster by cluster, see
#https://github.com/neurospin/brainomics-team/blob/master/2017_logistic_nestv/\
#scripts/deptms_pattern_interpretation.py

CLUSTER_LABELS_RH = "/neurospin/brainomics/2017_memento/analysis/FS/results/\
pcatv_FS_all/components/1e-4/struct_pca_0.01_0.5_0.1/beta_clust_labels_rh.gii"

CLUSTER_LABELS_LH = "/neurospin/brainomics/2017_memento/analysis/FS/results/\
pcatv_FS_all/components/1e-4/struct_pca_0.01_0.5_0.1/beta_clust_labels_lh.gii"

comp = components[:,1]
beta_mesh_r_label = gifti.read(CLUSTER_LABELS_RH).darrays[0].data
np.unique(beta_mesh_r_label)
beta_mesh_l_label = gifti.read(CLUSTER_LABELS_LH ).darrays[0].data
np.unique(beta_mesh_l_label)


beta_label = mesh_lr_to_beta(beta_mesh_l=beta_mesh_l_label, beta_mesh_r=beta_mesh_r_label+100, mask_mesh=mask_mesh)
len(np.unique(beta_label))
assert beta_label.shape[0] == X_hcp.shape[1]

#100 == 0 = not a clusters so same value in both hemisphere
beta_label[beta_label == 100] = 0
len(np.unique(beta_label))


rois_label2names={
   0: ["other", None],
   1:["Left postcentral",                  "Pos1"],
   2:["Left temporalpole",                 "Neg1"],
   3:["Left medialorbitofrontal",          "Neg2"],
   101:["Right temporalpole",              "Neg1"],
   102:["Right postcentral",               "Pos1"]}

# Extract a single score for each cluster
K = len(np.unique(beta_label))  # nb cluster
N = X_hcp.shape[0]
Xrois_avg = np.zeros((N, K))  # Xc . beta
Xrois_beta = np.zeros((N, K)) # mean(X_roi)
rois_beta_sign = np.zeros(K)
rois_beta_prop = np.zeros(K)

labels = np.unique(beta_label)
rois_names = [rois_label2names[int(k)][0] for k in labels]

for i, k in enumerate(labels):
    mask = beta_label == k
    Xrois_beta[:, i] = np.dot(X_hcp[:, mask], comp[mask]).ravel()
    Xrois_avg[:, i] = np.mean(X_hcp[:, mask], axis=1)
    rois_beta_sign[i] = np.sign(np.mean(comp[mask]))
    rois_beta_prop[i] = np.sum(comp[mask] ** 2) / np.sum(comp ** 2)
    print("Cluster:",k, "size:", mask.sum(), "\t", rois_names[i], np.mean(comp[mask]))
    #rois_beta_df_mean0[k] = np.mean(X[y==0, :][:, mask])
    #rois_beta_df_mean1[k] = np.mean(X[y==1, :][:, mask])

"""
Cluster: 0.0 size: 175797        other -0.0128050350703
Cluster: 1.0 size: 28218         Left postcentral 0.241016057676
Cluster: 2.0 size: 25901         Left temporalpole -0.232055363077
Cluster: 3.0 size: 6328          Left medialorbitofrontal -0.204149853502
Cluster: 101.0 size: 34849       Right temporalpole -0.231989101882
Cluster: 102.0 size: 28786       Right postcentral 0.248494359063
"""

rois_beta_df = pd.DataFrame(Xrois_beta)
rois_avg_df = pd.DataFrame(Xrois_avg)

rois_beta_df.columns = rois_names
rois_avg_df.columns = rois_names

rois_avg_df["GM_mean"] = X_hcp.mean(axis=1)

rois_info = pd.DataFrame(rois_names, columns=["roi"])
rois_info["sign"]  = rois_beta_sign
rois_info["prop"]  = rois_beta_prop
#rois_info["keep"] = rois_info["roi"].isin(rois_names_keep)


#plt.plot(Xrois_beta[:,1],Xrois_beta[:,2],'o')
plt.plot(Xrois_beta[:,1],Xrois_beta[:,2],'o',label = "CTL of hcp")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Left Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_hcp/postcentralL_vs_temporalL.png")

plt.plot(Xrois_avg[:,1],Xrois_avg[:,2],'o',label = "CTL of hcp")
plt.legend()
plt.xlabel("Left Postcentral")
plt.ylabel("Left Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_hcp/XpostcentralL_vs_temporalL.png")


plt.plot(Xrois_beta[:,5],Xrois_beta[:,4],'o',label = "CTL of hcp")
plt.legend()
plt.xlabel("score on Right Postcentral ")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_hcp/postcentralR_vs_temporalR.png")




###########################################################################
#Correlation acoording to age
age = pop_hcp["Age"].values


x0 = Xrois_beta[(age<25),5]
y0 = Xrois_beta[(age<25),4]
fit = np.polyfit(x0, y0, deg=1)
plt.plot(x0, fit[0] * x0 + fit[1],label = "<25")
plt.scatter(x0,y0,color = "b")
x1 = Xrois_beta[(age>25) & (age<30),5]
y1 = Xrois_beta[(age>25) & (age<30),4]
fit1 = np.polyfit(x1, y1, deg=1)
plt.plot(x1, fit1[0] * x1 + fit1[1],label = "between 25 and 30")
plt.scatter(x1,y1,color = "g")
x2 = Xrois_beta[(age>30) & (age<70),5]
y2 = Xrois_beta[(age>30) & (age<70),4]
fit2 = np.polyfit(x2, y2, deg=1)
plt.plot(x2, fit2[0] * x2 + fit2[1],label = ">30")
plt.scatter(x2,y2,color = "r")
plt.legend()
plt.xlabel("score on Right Postcentral ")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/clusters_comp1_hcp/tempR_vs_RpostCentral.png")








#mri_surfcluster --in beta_lh.gii\
# --hemi lh --subject fsaverage\
# --thmin  0.0001\
# --sum ./beta_clust_info_lh.txt\
# --annot aparc\
# --o  ./beta_clust_values_lh.gii\
# --ocn  ./beta_clust_labels_lh.gii
#mri_surfcluster --in beta_rh.gii\
# --hemi rh --subject fsaverage\
# --thmin  0.0001\
# --sum ./beta_clust_info_rh.txt\
# --annot aparc\
# --o  ./beta_clust_values_rh.gii\
# --ocn  ./beta_clust_labels_rh.gii
#
#
#
#
#
def mesh_lr_to_beta(beta_mesh_l, beta_mesh_r, mask_mesh):
    beta_mesh = np.zeros(mask_mesh.shape)
    idx_r = int(beta_mesh.shape[0] / 2)
    beta_mesh[:idx_r] = beta_mesh_l
    beta_mesh[idx_r:] = beta_mesh_r
    beta = beta_mesh[mask_mesh]
    return beta
