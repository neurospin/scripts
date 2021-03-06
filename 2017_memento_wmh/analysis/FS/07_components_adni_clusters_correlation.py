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
#Project ADNI subjects
INPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/data/adni/population.csv"
pop_adni  = pd.read_csv(INPUT_CSV)

X_adni = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/adni/X.npy")
y_adni = np.load("/neurospin/brainomics/2017_memento/analysis/FS/data/adni/y.npy").ravel()
assert X_adni.shape == (706, 299879)

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
assert beta_label.shape[0] == X_adni.shape[1]

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
N = X_adni.shape[0]
Xrois_avg = np.zeros((N, K))  # Xc . beta
Xrois_beta = np.zeros((N, K)) # mean(X_roi)
rois_beta_sign = np.zeros(K)
rois_beta_prop = np.zeros(K)

labels = np.unique(beta_label)
rois_names = [rois_label2names[int(k)][0] for k in labels]

for i, k in enumerate(labels):
    mask = beta_label == k
    Xrois_beta[:, i] = np.dot(X_adni[:, mask], comp[mask]).ravel()
    Xrois_avg[:, i] = np.mean(X_adni[:, mask], axis=1)
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

rois_avg_df["GM_mean"] = X_adni.mean(axis=1)

rois_info = pd.DataFrame(rois_names, columns=["roi"])
rois_info["sign"]  = rois_beta_sign
rois_info["prop"]  = rois_beta_prop
#rois_info["keep"] = rois_info["roi"].isin(rois_names_keep)

WD_CLUST = "/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/clusters_comp1_adni"
with pd.ExcelWriter(os.path.join(WD_CLUST, "rois.xlsx")) as writer:
    rois_beta_df.to_excel(writer, sheet_name='roi_beta', index=False)
    rois_info.to_excel(writer, sheet_name='roi_info', index=False)
    rois_avg_df.to_excel(writer, sheet_name='roi_avg', index=False)

#Correlation of each cluster with age
###############################################################################
age = pop_adni['Age at inclusion']
plt.plot(age[y_adni==0],Xrois_beta[y_adni==0,2],'o',label = "CTL")
plt.plot(age[y_adni==1],Xrois_beta[y_adni==1,2],'o',label = "MCInc")
plt.plot(age[y_adni==2],Xrois_beta[y_adni==2,2],'o',label = "MCIc")
plt.plot(age[y_adni==3],Xrois_beta[y_adni==3,2],'o',label = "AD")
plt.xlabel("age")
plt.ylabel("score on Left Temporalpole")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/age/Ltemporal_vs_age.png")

age = pop_adni['Age at inclusion']
plt.plot(age[y_adni==0],Xrois_beta[y_adni==0,1],'o',label = "CTL")
plt.plot(age[y_adni==1],Xrois_beta[y_adni==1,1],'o',label = "MCInc")
plt.plot(age[y_adni==2],Xrois_beta[y_adni==2,1],'o',label = "MCIc")
plt.plot(age[y_adni==3],Xrois_beta[y_adni==3,1],'o',label = "AD")
plt.xlabel("age")
plt.ylabel("score on Left Postcentral")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/age/LPostcentral_vs_age.png")

age = pop_adni['Age at inclusion']
plt.plot(age[y_adni==0],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.plot(age[y_adni==1],Xrois_beta[y_adni==1,4],'o',label = "MCInc")
plt.plot(age[y_adni==2],Xrois_beta[y_adni==2,4],'o',label = "MCIc")
plt.plot(age[y_adni==3],Xrois_beta[y_adni==3,4],'o',label = "AD")
plt.xlabel("age")
plt.ylabel("score on Right Temporal")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/age/Rtemporal_vs_age.png")

age = pop_adni['Age at inclusion']
plt.plot(age[y_adni==0],Xrois_beta[y_adni==0,5],'o',label = "CTL")
plt.plot(age[y_adni==1],Xrois_beta[y_adni==1,5],'o',label = "MCInc")
plt.plot(age[y_adni==2],Xrois_beta[y_adni==2,5],'o',label = "MCIc")
plt.plot(age[y_adni==3],Xrois_beta[y_adni==3,5],'o',label = "AD")
plt.xlabel("age")
plt.ylabel("score on Right Postcentral")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/age/RPostcentral_vs_age.png")


#Correlation between clusters
###############################################################################

#plt.plot(Xrois_beta[:,1],Xrois_beta[:,2],'o')
plt.plot(Xrois_beta[y_adni==0,1],Xrois_beta[y_adni==0,2],'o',label = "CTL")
plt.plot(Xrois_beta[y_adni==1,1],Xrois_beta[y_adni==1,2],'o',label = "MCInc")
plt.plot(Xrois_beta[y_adni==2,1],Xrois_beta[y_adni==2,2],'o',label = "MCIc")
plt.plot(Xrois_beta[y_adni==3,1],Xrois_beta[y_adni==3,2],'o',label = "AD")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Left Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentralL_vs_temporalL.png")


x0 = Xrois_beta[y_adni==0,1]
y0 = Xrois_beta[y_adni==0,2]
fit0 = np.polyfit(x0, y0, deg=1)
plt.plot(x0, fit0[0] * x0 + fit0[1],label = "CTL",color = "b")
plt.scatter(x0,y0,color = "b")
x1 = Xrois_beta[y_adni==1,1]
y1 = Xrois_beta[y_adni==1,2]
fit1 = np.polyfit(x1, y1, deg=1)
plt.plot(x1, fit1[0] * x1 + fit1[1],label = "MCInc",color = "g")
plt.scatter(x1,y1,color = "g")
x2 = Xrois_beta[y_adni==2,1]
y2 = Xrois_beta[y_adni==2,2]
fit2 = np.polyfit(x2, y2, deg=1)
plt.plot(x0, fit2[0] * x0 + fit2[1],label = "MCIc",color = "r")
plt.scatter(x2,y2,color = "r")
x3 = Xrois_beta[y_adni==3,1]
y3 = Xrois_beta[y_adni==3,2]
fit3 = np.polyfit(x3, y3, deg=1)
plt.plot(x3, fit3[0] * x3 + fit3[1],label = "AD",color = "m")
plt.scatter(x3,y3,color = "m")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/postcentralL_vs_temporalL_with_fit.png")

plt.plot(Xrois_beta[y_adni==0,1],Xrois_beta[y_adni==0,2],'o',label = "CTL")
plt.plot(Xrois_beta[y_adni==1,1],Xrois_beta[y_adni==1,2],'o',label = "MCInc")
plt.plot(Xrois_beta[y_adni==2,1],Xrois_beta[y_adni==2,2],'o',label = "MCIc")
plt.plot(Xrois_beta[y_adni==3,1],Xrois_beta[y_adni==3,2],'o',label = "AD")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Left Temporalpole")





plt.plot(Xrois_beta[y_adni==0,1],Xrois_beta[y_adni==0,2],'o',label = "CTL")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Left Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentralL_vs_temporalL_CTL.png")

plt.plot(Xrois_beta[y_adni==0,5],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.plot(Xrois_beta[y_adni==1,5],Xrois_beta[y_adni==1,4],'o',label = "MCInc")
plt.plot(Xrois_beta[y_adni==2,5],Xrois_beta[y_adni==2,4],'o',label = "MCIc")
plt.plot(Xrois_beta[y_adni==3,5],Xrois_beta[y_adni==3,4],'o',label = "AD")
plt.legend()
plt.xlabel("score on Right Postcentral ")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentralR_vs_temporalR.png")


plt.plot(Xrois_beta[y_adni==0,5],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.legend()
plt.xlabel("score on Right Postcentral ")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentralR_vs_temporalR_CTL.png")


x0 = Xrois_beta[y_adni==0,5]
y0 = Xrois_beta[y_adni==0,4]
fit0 = np.polyfit(x0, y0, deg=1)
plt.plot(x0, fit0[0] * x0 + fit0[1],label = "CTL",color = "b")
plt.scatter(x0,y0,color = "b")
x1 = Xrois_beta[y_adni==1,5]
y1 = Xrois_beta[y_adni==1,4]
fit1 = np.polyfit(x1, y1, deg=1)
plt.plot(x1, fit1[0] * x1 + fit1[1],label = "MCInc",color = "g")
plt.scatter(x1,y1,color = "g")
x2 = Xrois_beta[y_adni==2,5]
y2 = Xrois_beta[y_adni==2,4]
fit2 = np.polyfit(x2, y2, deg=1)
plt.plot(x0, fit2[0] * x0 + fit2[1],label = "MCIc",color = "r")
plt.scatter(x2,y2,color = "r")
x3 = Xrois_beta[y_adni==3,5]
y3 = Xrois_beta[y_adni==3,4]
fit3 = np.polyfit(x3, y3, deg=1)
plt.plot(x3, fit3[0] * x3 + fit3[1],label = "AD",color = "m")
plt.scatter(x3,y3,color = "m")
plt.legend()
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/postcentralR_vs_temporalR_with_fit.png")

plt.plot(Xrois_beta[y_adni==0,2],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.plot(Xrois_beta[y_adni==1,2],Xrois_beta[y_adni==1,4],'o',label = "MCInc")
plt.plot(Xrois_beta[y_adni==2,2],Xrois_beta[y_adni==2,4],'o',label = "MCIc")
plt.plot(Xrois_beta[y_adni==3,2],Xrois_beta[y_adni==3,4],'o',label = "AD")
plt.legend()
plt.xlabel("score on Left Temporalpole")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/temporal_r_vs_l.png")

plt.plot(Xrois_beta[y_adni==0,2],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.legend()
plt.xlabel("score on Left Temporalpole")
plt.ylabel("score on Right Temporalpole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/temporal_r_vs_l_CTL.png")


plt.plot(Xrois_beta[y_adni==0,1],Xrois_beta[y_adni==0,5],'o',label = "CTL")
plt.plot(Xrois_beta[y_adni==1,1],Xrois_beta[y_adni==1,5],'o',label = "MCInc")
plt.plot(Xrois_beta[y_adni==2,1],Xrois_beta[y_adni==2,5],'o',label = "MCIc")
plt.plot(Xrois_beta[y_adni==3,1],Xrois_beta[y_adni==3,5],'o',label = "AD")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Right Postcentral")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentral_r_vs_l.png")


plt.plot(Xrois_beta[y_adni==0,1],Xrois_beta[y_adni==0,5],'o',label = "CTL")
plt.legend()
plt.xlabel("score on Left Postcentral")
plt.ylabel("score on Right Postcentral")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1/postcentral_r_vs_l_CTL.png")


#################################################################################

#Correlation with clinical scores

plt.plot(pop_adni["ADAS11.sc"][y_adni==0],Xrois_beta[y_adni==0,5],'o',label = "CTL")
plt.plot(pop_adni["ADAS11.sc"][y_adni==1],Xrois_beta[y_adni==1,5],'o',label = "MCInc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==2],Xrois_beta[y_adni==2,5],'o',label = "MCIc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==3],Xrois_beta[y_adni==3,5],'o',label = "AD")
plt.legend()
x = pop_adni["ADAS11.sc"][np.isnan(pop_adni["ADAS11.sc"])==False]
y = Xrois_beta[:,5][np.array(np.isnan(pop_adni["ADAS11.sc"])==False)]
Coef,pvalue = scipy.stats.pearsonr(x,y)
plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
plt.xlabel("ADAS_score")
plt.ylabel("score on Right Postcentral")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/adas_postcentralR.png")


plt.plot(pop_adni["ADAS11.sc"][y_adni==0],Xrois_beta[y_adni==0,1],'o',label = "CTL")
plt.plot(pop_adni["ADAS11.sc"][y_adni==1],Xrois_beta[y_adni==1,1],'o',label = "MCInc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==2],Xrois_beta[y_adni==2,1],'o',label = "MCIc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==3],Xrois_beta[y_adni==3,1],'o',label = "AD")
plt.legend()
x = pop_adni["ADAS11.sc"][np.isnan(pop_adni["ADAS11.sc"])==False]
y = Xrois_beta[:,1][np.array(np.isnan(pop_adni["ADAS11.sc"])==False)]
Coef,pvalue = scipy.stats.pearsonr(x,y)
plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
plt.xlabel("ADAS_score")
plt.ylabel("score on Left Postcentral")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/adas_postcentralL.png")


plt.plot(pop_adni["ADAS11.sc"][y_adni==0],Xrois_beta[y_adni==0,2],'o',label = "CTL")
plt.plot(pop_adni["ADAS11.sc"][y_adni==1],Xrois_beta[y_adni==1,2],'o',label = "MCInc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==2],Xrois_beta[y_adni==2,2],'o',label = "MCIc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==3],Xrois_beta[y_adni==3,2],'o',label = "AD")
plt.legend()
x = pop_adni["ADAS11.sc"][np.isnan(pop_adni["ADAS11.sc"])==False]
y = Xrois_beta[:,2][np.array(np.isnan(pop_adni["ADAS11.sc"])==False)]
Coef,pvalue = scipy.stats.pearsonr(x,y)
plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
plt.xlabel("ADAS_score")
plt.ylabel("score on Left temporal pole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/adas_temporalL.png")


plt.plot(pop_adni["ADAS11.sc"][y_adni==0],Xrois_beta[y_adni==0,4],'o',label = "CTL")
plt.plot(pop_adni["ADAS11.sc"][y_adni==1],Xrois_beta[y_adni==1,4],'o',label = "MCInc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==2],Xrois_beta[y_adni==2,4],'o',label = "MCIc")
plt.plot(pop_adni["ADAS11.sc"][y_adni==3],Xrois_beta[y_adni==3,4],'o',label = "AD")
plt.legend()
x = pop_adni["ADAS11.sc"][np.isnan(pop_adni["ADAS11.sc"])==False]
y = Xrois_beta[:,4][np.array(np.isnan(pop_adni["ADAS11.sc"])==False)]
Coef,pvalue = scipy.stats.pearsonr(x,y)
plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
plt.xlabel("ADAS_score")
plt.ylabel("score on Right temporal pole")
plt.savefig("/neurospin/brainomics/2017_memento/analysis/FS/results/pcatv_FS_all/\
clusters_comp1_adni/adas_temporalR.png")

## -------------------------------
## Group ROIs correlation matrix
## -------------------------------
import seaborn as sns

roi_xls_filename = os.path.join(WD_CLUST, "rois.xlsx")
rois_beta_df = pd.read_excel(roi_xls_filename, sheetname='roi_beta')
rois_avg_df = pd.read_excel(roi_xls_filename, sheetname='roi_avg')
rois_info_df = pd.read_excel(roi_xls_filename, sheetname='roi_info')

rois_beta_df["GM_mean"] = rois_avg_df["GM_mean"]

##
# Seaborn
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp

# https://stackoverflow.com/questions/38705359/how-to-give-sns-clustermap-a-precomputed-distance-matrix
DF = rois_beta_df.copy()

DF_corr = DF.corr()
DF_dism = 1 - DF_corr# ** 2

linkage = hc.linkage(sp.distance.squareform(DF_dism), method='average')
g = sns.clustermap(DF_corr,  col_linkage=linkage, row_linkage=linkage)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
plt.savefig(os.path.join(WD_CLUST, "cor_rois.pdf"))

# Positive / Negatives
# --------------------

beta_pos_msk = comp > 0
beta_neg_msk = comp < 0

print(beta_pos_msk.sum(), beta_neg_msk.sum())


Xscores = np.zeros((X_adni.shape[0], 5))

Xscores[:, 0] = np.dot(X_adni[:, beta_pos_msk], comp[beta_pos_msk]).ravel()
Xscores[:, 1] = X_adni[:, beta_pos_msk].mean(axis=1)

Xscores[:, 2] = -np.dot(X_adni[:, beta_neg_msk], comp[beta_neg_msk]).ravel()
Xscores[:, 3] = X_adni[:, beta_neg_msk].mean(axis=1)

Xscores[:, 4] = X_adni.mean(axis=1)

scores = pd.DataFrame(Xscores,
                      columns=['xbeta_pos', 'xavg_pos', '-xbeta_neg', 'xavg_neg', 'xavg'])

corr = scores.corr()
f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, cmap=plt.cm.bwr, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(WD_CLUST, "cor_rois_raw.pdf"))

"""
           xbeta_pos  xbeta_neg  xavg_pos  xavg_neg      xavg
xbeta_pos   1.000000   0.438476  0.933818 -0.401512  0.285163
xbeta_neg   0.438476   1.000000  0.477728 -0.976621 -0.301655
xavg_pos    0.933818   0.477728  1.000000 -0.442388  0.353100
xavg_neg   -0.401512  -0.976621 -0.442388  1.000000  0.342208
xavg        0.285163  -0.301655  0.353100  0.342208  1.000000
"""
scores["DX"] = y_adni

#plt.scatter(scores[], scores[], color=scores["y"])

fig = plt.figure(figsize=(20, 20))
g = sns.PairGrid(scores, hue="DX")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(WD_CLUST, "rois_scatterplot.pdf"))

from sklearn.decomposition import PCA
# PCA with scikit-learn
pca = PCA(n_components=2)
pca.fit(Xscores[:, [0, 2]])
print(pca.explained_variance_ratio_)

scores["pc1_xbeta_pos_neg"] = pca.transform(Xscores[:, [0, 2]])[:, 0]
scores["pc2_xbeta_pos_neg"] = pca.transform(Xscores[:, [0, 2]])[:, 1]

np.corrcoef(scores["xavg"], scores["pc1_xbeta_pos_neg"])
np.corrcoef(scores["xavg"], scores["pc2_xbeta_pos_neg"])



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
