#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:35:27 2017

@author: ad247405
"""
import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns, matplotlib.pyplot as plt
import scipy.stats
import nibabel
import scipy.stats

INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/X.npy"

INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/y.npy"

POPULATION_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"

pop = pd.read_csv(POPULATION_CSV)
age = pop["age"].values



X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y).ravel()
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")


SAPS_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nmorphch.npy")
SANS_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nmorphch.npy")

X_nmorph = X[site==2,:]
y_nmorph = y[site==2]
X_nmorph_scz = X_nmorph[y_nmorph==1,:]
X_nmorph_scz = X_nmorph_scz[np.logical_not(np.isnan(SANS_nmorph))]
age_nmorph = age[site==2]
age_nmorph_scz = age_nmorph[y_nmorph==1]
age_nmorph_scz = age_nmorph_scz[np.logical_not(np.isnan(SANS_nmorph))]

age_scz = age_nmorph[y_nmorph==1]
age_con = age_nmorph[y_nmorph==0]

SAPS_nmorph = SAPS_nmorph[np.logical_not(np.isnan(SANS_nmorph))]
SANS_nmorph = SANS_nmorph[np.logical_not(np.isnan(SANS_nmorph))]
assert SAPS_nmorph.shape == SANS_nmorph.shape == (41,)





MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii"
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]


CLUSTER_LABELS = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/enetall_all+VIP_all/5cv/refit/refit/\
enettv_0.1_0.1_0.8/weight_map_clust_labels.nii.gz"


labels_img  = nibabel.load(CLUSTER_LABELS)
labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]

N_all= X_nmorph.shape[0]
N = X_nmorph_scz.shape[0]

# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
scores_nmorph_scz = np.zeros((N, K))
scores_nmorph_all = np.zeros((N_all, K))

K_interest = [18,14,33,20,4,25,23,22,15,41]

for k in K_interest:
    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores_nmorph_scz[:, k] = np.dot(X_nmorph_scz[:, mask], beta[mask]).ravel()
    scores_nmorph_all[:, k] = np.dot(X_nmorph[:, mask], beta[mask]).ravel()




#Plot PANSS correlation
for i in K_interest:
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
    corr,p = scipy.stats.pearsonr(scores_nmorph_scz[:,i],SAPS_nmorph)
    ax1.plot(scores_nmorph_scz[:,i] ,SAPS_nmorph,'o')
    ax1.set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    ax1.set_xlabel('Score on cluster %s' %(i))
    ax1.set_ylabel('SAPS score')
    corr,p = scipy.stats.pearsonr(scores_nmorph_scz[:,i],SANS_nmorph)
    ax2.plot(scores_nmorph_scz[:,i] ,SANS_nmorph,'o')
    ax2.set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    ax2.set_xlabel('Score on cluster %s' %(i))
    ax2.set_ylabel('SANS score')
    fig.tight_layout()
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/nmorph_cluster%s" %(i))



#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_nmorph_all[:,i],age_nmorph)
    plt.figure()
    plt.plot(scores_nmorph_all[y_nmorph==0,i] ,age_con,'o',label = "controls")
    plt.plot(scores_nmorph_all[y_nmorph==1,i] ,age_scz,'o',label = "patients")
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('age')
    plt.tight_layout()
    plt.legend(fontsize = 15,loc = "upper left")
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/age_corr/nmorph_cluster_age%s" %(i))


