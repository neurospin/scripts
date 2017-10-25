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
import sklearn

INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/X.npy"

INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/y.npy"

penalty_start = 2
INPUT_POP = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv"
pop = pd.read_csv(INPUT_POP)
age = pop.age.values




X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y).ravel()
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")

age_scz = age[y==1]
age_con = age[y==0]
X_con = X[y==0,:]
X_scz = X[y==1,:]

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

N = X.shape[0]


# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
scores = np.zeros((N, K))


K_interest = [18,14,33,20,4,25,23,22,15,41]

for k in K_interest:
    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores[:, k] = np.dot(X[:, mask], beta[mask]).ravel()




#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores[:,i],age)
    plt.figure()
    plt.plot(scores[y==0,i] ,age_con,'o',label = "controls")
    plt.plot(scores[y==1,i] ,age_scz,'o',label = "patients")
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('age')
    plt.tight_layout()
    plt.legend(fontsize = 15,loc = "upper left")
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/age_corr/cluster_age%s" %(i))




    ###########################################################################



#clusts = dict(Cingulate = 18, Caudate = 14, Precentral = 33 , Paracingulate = 20)
clusts = dict(Cingulate = 18, RightCaudate = 14, Precentral = 33 , Paracingulate = 20,\
              TemporalPole= 4, LeftHyppocampusAmygdala = 25, LeftCaudate=23, LeftThalamus = 22,\
              RightThalamus = 15,MiddleTemporalGyrus = 41)

clusts = dict(Cingulate = 18, RightCaudate = 14, Precentral = 33 , Paracingulate = 20,\
              TemporalPole= 4)


clust_oi = [clusts[k] for k in clusts]

Dp = pd.DataFrame(np.concatenate([scores[:, clust_oi], y[:, np.newaxis]], axis=1), columns=list(clusts.keys())+["Condition"])

Dp['Age'] = age

g = sns.PairGrid(Dp, hue="Condition")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/ROIS_correlations.png")


##############################################################################

