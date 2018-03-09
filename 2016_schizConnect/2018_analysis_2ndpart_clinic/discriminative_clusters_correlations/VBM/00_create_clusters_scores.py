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


MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii"
babel_mask  = nb.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()



WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]


CLUSTER_LABELS = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/enetall_all+VIP_all/5cv/refit/refit/\
enettv_0.1_0.1_0.8/weight_map_clust_labels.nii.gz"

labels_img  = nb.load(CLUSTER_LABELS)
labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]

#Save all subjects
pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
pop_all_scz = pop_all[pop_all['dx_num']==1]
pop_all_con = pop_all[pop_all['dx_num']==0]
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
assert X_all.shape == (606, 125961)
X_all_scz = X_all[y_all==1,2:]
X_all_con = X_all[y_all==0,2:]
X_all = X_all[:,2:]

assert X_all_scz.shape == (276, 125959)

N_scz = X_all_scz.shape[0]
N_con = X_all_con.shape[0]
N = X_all.shape[0]

# Extract a single score for each cluster
K_interest = [18,14,33,20,4,25,23,22,15,41]
scores_all_scz = np.zeros((N_scz, len(K_interest)+1))
scores_all_con = np.zeros((N_con, len(K_interest)+1))
scores_all = np.zeros((N, len(K_interest)+1))

i=0
for k in range (len(K_interest)):

    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores_all_scz[:, i] = np.dot(X_all_scz[:, mask], beta[mask]).ravel()
    scores_all_con[:, i] = np.dot(X_all_con[:, mask], beta[mask]).ravel()
    scores_all[:, i] = np.dot(X_all[:, mask], beta[mask]).ravel()
    i= i+1

mask = labels_flt == (18 or  14 or 33 or 20 or 4 or 25 or 23 or 22 or 15 or 41)
#mask = labels_flt == (18 or 4 or 25)
scores_all_scz[:, 10] = np.dot(X_all_scz[:, mask], beta[mask]).ravel()
scores_all_con[:, 10] = np.dot(X_all_con[:, mask], beta[mask]).ravel()
scores_all[:, 10] = np.dot(X_all[:,mask], beta[mask]).ravel()

pop_all_scz["cluster1_cingulate_gyrus"] = scores_all_scz[:, 0]
pop_all_scz["cluster2_right_caudate_putamen"] = scores_all_scz[:,1]
pop_all_scz["cluster3_precentral_postcentral_gyrus"] = scores_all_scz[:, 2]
pop_all_scz["cluster4_frontal_pole"] = scores_all_scz[:, 3]
pop_all_scz["cluster5_temporal_pole"] = scores_all_scz[:, 4]
pop_all_scz["cluster6_left_hippocampus_amygdala"] = scores_all_scz[:, 5]
pop_all_scz["cluster7_left_caudate_putamen"] = scores_all_scz[:, 6]
pop_all_scz["cluster8_left_thalamus"] = scores_all_scz[:, 7]
pop_all_scz["cluster9_right_thalamus"] = scores_all_scz[:, 8]
pop_all_scz["cluster10_middle_temporal_gyrus"] = scores_all_scz[:, 9]
pop_all_scz["cluster11_predictive_signature"] = scores_all_scz[:, 10]

pop_all["cluster1_cingulate_gyrus"] = scores_all[:, 0]
pop_all["cluster2_right_caudate_putamen"] = scores_all[:,1]
pop_all["cluster3_precentral_postcentral_gyrus"] = scores_all[:, 2]
pop_all["cluster4_frontal_pole"] = scores_all[:, 3]
pop_all["cluster5_temporal_pole"] = scores_all[:, 4]
pop_all["cluster6_left_hippocampus_amygdala"] = scores_all[:, 5]
pop_all["cluster7_left_caudate_putamen"] = scores_all[:, 6]
pop_all["cluster8_left_thalamus"] = scores_all[:, 7]
pop_all["cluster9_right_thalamus"] = scores_all[:, 8]
pop_all["cluster10_middle_temporal_gyrus"] = scores_all[:, 9]
pop_all["cluster11_predictive_signature"] = scores_all[:, 10]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
pop_all_scz.to_csv(os.path.join(output,"pop_all_scz.csv") , index=False)

pop_cobre_scz = pop_all_scz[pop_all_scz["site_num"]==1]
pop_cobre_scz.to_csv(os.path.join(output,"pop_cobre_scz.csv") , index=False)

pop_nmorph_scz = pop_all_scz[pop_all_scz["site_num"]==2]
pop_nmorph_scz.to_csv(os.path.join(output,"pop_nmorph_scz.csv") , index=False)

pop_nudast_scz = pop_all_scz[pop_all_scz["site_num"]==3]
pop_nudast = pop_all[pop_all["site_num"]==3]
pop_nudast_scz.to_csv(os.path.join(output,"pop_nudast_scz.csv") , index=False)
pop_nudast.to_csv(os.path.join(output,"pop_nudast.csv") , index=False)

pop_vip_scz = pop_all_scz[pop_all_scz["site_num"]==4]
pop_vip_scz.to_csv(os.path.join(output,"pop_vip_scz.csv") , index=False)


#Test discriminative power of each cluster with a paired t test
##############################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/clusters_ttest"
for i in range(11):
    plt.figure()
    df = pd.DataFrame()
    df["score"] = scores_all[:,i]
    df["dx"] = y_all
    T, p = scipy.stats.ttest_ind(scores_all_scz[:, i],scores_all_con[:, i])
    print("Cluster %s: T = %s and p = %s" %(i,T,p))
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%(i+1))
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),p)))
    plt.savefig(os.path.join(output,"cluster%s"%((i+1))))

