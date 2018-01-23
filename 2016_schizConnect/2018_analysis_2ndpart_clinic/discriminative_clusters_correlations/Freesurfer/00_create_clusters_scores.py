#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:08:33 2017

@author: ad247405
"""


import os
import subprocess
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
import brainomics.array_utils
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from collections import OrderedDict
import nilearn
from nilearn import plotting
from nilearn import image
import seaborn as sns
import matplotlib.pylab as plt
import shutil
import sys
sys.path.insert(0,'/home/ed203246/git/scripts/brainomics')
import array_utils, mesh_processing
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/\
all_subjects/results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][:]



pop_all_scz = pop_all[pop_all['dx_num']==1]
pop_all_con = pop_all[pop_all['dx_num']==0]
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/X.npy")
assert X_all.shape == (567, 299865)
X_all_scz = X_all[y_all==1,:]
X_all_con = X_all[y_all==0,:]

assert X_all_scz.shape == (253, 299865)

N_scz = X_all_scz.shape[0]
N_con = X_all_con.shape[0]
N = X_all.shape[0]

mask_mesh = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/mask.npy")

beta, t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
beta = beta.ravel()

assert X_all_scz.shape[1] == beta.shape[0]
N = X_all_scz.shape[0]
print(pd.Series(beta.ravel()).describe(), t)

# Write morph data
from nibabel import gifti


[coords_l, faces_l], beta_mesh_l, [coords_r, faces_r], beta_mesh_r, stat = \
    beta_to_mesh_lr(beta, mask_mesh, mesh_l, mesh_r, threshold=1.)

hemi, view = 'right', 'medial'

if hemi == 'left':
    coords_x, faces_x, beta_mesh_x, sulc_x =\
        coords_l, faces_l, beta_mesh_l, sulc_l
elif hemi == 'right':
    coords_x, faces_x, beta_mesh_x, sulc_x =\
        coords_r, faces_r, beta_mesh_r, sulc_r

vmax_beta = np.max(np.abs(beta)) / 10
vmax_beta = np.max(np.abs(beta_mesh_x) * 1000) / 10

plotting.plot_surf_stat_map([coords_x, faces_x], stat_map=1000 * beta_mesh_x,
                            hemi=hemi, view=view,
                            bg_map=sulc_x, #bg_on_data=True,
                            #vmax = vmax_beta,#stat[2] / 10,#vmax=vmax_beta,
                            darkness=.5,
                            cmap=plt.cm.seismic,
                            #symmetric_cbar=True,
                            #output_file=output_filename
                            )

print(pd.Series((beta_mesh_l[beta_mesh_l != 0]).ravel()).describe())
print(pd.Series((beta_mesh_r[beta_mesh_r != 0]).ravel()).describe())

mesh_processing.save_texture(os.path.join(WD_CLUST, "beta_lh.gii"),
                             beta_mesh_l)

mesh_processing.save_texture(os.path.join(WD_CLUST, "beta_rh.gii"),
                             beta_mesh_r)

# Extract a single score for each cluster
K_interest = [18,14,33,20,4,25,23,22,15,41]
scores_all_scz = np.zeros((N_scz, len(K_interest)))
scores_all_con = np.zeros((N_con, len(K_interest)))
scores_all = np.zeros((N, len(K_interest)))

i=0
for k in range (len(K_interest)):

    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores_all_scz[:, i] = np.dot(X_all_scz[:, mask], beta[mask]).ravel()
    scores_all_con[:, i] = np.dot(X_all_con[:, mask], beta[mask]).ravel()
    scores_all[:, i] = np.dot(X_all[:, mask], beta[mask]).ravel()
    i= i+1



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


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
pop_all_scz.to_csv(os.path.join(output,"pop_all_scz.csv") , index=False)

pop_cobre_scz = pop_all_scz[pop_all_scz["site_num"]==1]
pop_cobre_scz.to_csv(os.path.join(output,"pop_cobre_scz.csv") , index=False)

pop_nmorph_scz = pop_all_scz[pop_all_scz["site_num"]==2]
pop_nmorph_scz.to_csv(os.path.join(output,"pop_nmorph_scz.csv") , index=False)

pop_nudast_scz = pop_all_scz[pop_all_scz["site_num"]==3]
pop_nudast_scz.to_csv(os.path.join(output,"pop_nudast_scz.csv") , index=False)

pop_vip_scz = pop_all_scz[pop_all_scz["site_num"]==4]
pop_vip_scz.to_csv(os.path.join(output,"pop_vip_scz.csv") , index=False)


#Test discriminative power of each cluster with a paired t test
##############################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/clusters_ttest"
for i in range(10):
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




def beta_to_mesh_lr(beta, mask_mesh, mesh_l, mesh_r, threshold=.99):
    # beta to array of mesh size
    #ouput_filename = os.path.splitext(beta_filename)[0] + ".nii.gz"
    assert beta.shape[0] == mask_mesh.sum()
    beta_t, t = array_utils.arr_threshold_from_norm2_ratio(beta, threshold)
    #print(np.sum(beta != 0), np.sum(beta_t != 0), np.max(np.abs(beta_t)))
    beta_mesh = np.zeros(mask_mesh.shape)
    beta_mesh[mask_mesh] = beta_t.ravel()

    # mesh, l+r
    mesh_l = nilearn.plotting.surf_plotting.load_surf_mesh(mesh_l)
    coords_l, faces_l = mesh_l[0], mesh_l[1]
    mesh_r = nilearn.plotting.surf_plotting.load_surf_mesh(mesh_r)
    coords_r, faces_r = mesh_r[0], mesh_r[1]
    assert coords_l.shape[0] == coords_r.shape[0] == beta_mesh.shape[0] / 2

    beta_mesh_l = np.zeros(coords_l.shape)
    beta_mesh_l = beta_mesh[:coords_l.shape[0]]
    beta_mesh_r = np.zeros(coords_r.shape)
    beta_mesh_r = beta_mesh[coords_l.shape[0]:]

    return [coords_l, faces_l], beta_mesh_l, [coords_r, faces_r], beta_mesh_r, [np.sum(beta != 0), np.sum(beta_t != 0), np.max(np.abs(beta_t))]


def mesh_lr_to_beta(beta_mesh_l, beta_mesh_r, mask_mesh):
    beta_mesh = np.zeros(mask_mesh.shape)
    idx_r = int(beta_mesh.shape[0] / 2)
    beta_mesh[:idx_r] = beta_mesh_l
    beta_mesh[idx_r:] = beta_mesh_r
    beta = beta_mesh[mask_mesh]
    return beta
