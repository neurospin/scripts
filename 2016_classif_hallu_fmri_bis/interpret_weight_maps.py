#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:22:58 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import nilearn  
from nilearn import plotting
import matplotlib.pylab as plt
import pandas as pd
import brainomics.image_atlas
import brainomics.array_utils
import subprocess
import json
from scipy.stats.stats import pearsonr 

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
DATA_X = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/T.npy"
DATA_y = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/y_state.npy"


babel_mask  = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()


#Enet-TV
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6'


age = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_with_covariates/multivariate_analysis/data/T.npy')[:,0]


wm_folder = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6/"
labels_img  = nibabel.load(os.path.join(wm_folder,"weight_map_clust_labels.nii.gz"))

labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]


X = np.load(DATA_X)
y = np.load(DATA_y)
assert X.shape[1] == beta.shape[0] 
assert labels_flt.shape[0] == beta.shape[0]
N = X.shape[0]


# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
Xp = np.zeros((N, K))


for k in np.unique(labels_flt):
     mask = labels_flt == k
     print("Cluster:",k, "size:", mask.sum())
     Xp[:, k] = np.dot(X[:, mask], beta[mask]).ravel()
 
mask_cluster_right = labels_flt == 1
mask_cluster_left = labels_flt == 2    


#plot with the weight map refitted on all samples 
################################################################################     
plt.scatter(Xp[y==0, 1], Xp[y==0, 2], c='b', label='off - resting state')
plt.scatter(Xp[y==1, 1], Xp[y==1, 2], c='r', label='transition toward hallucination')
plt.xlabel("right cluster")
plt.ylabel("left cluster")
plt.legend(loc = "upper left")
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","clusters_correlation.png"))
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","clusters_correlation.pdf"))


plt.scatter(Xp[y==0, 1],age[y==0], c='b', label='off - resting state')
plt.scatter(Xp[y==1, 1], age[y==1],  c='r', label='off - resting state')
plt.xlabel("right cluster")
plt.ylabel("age")
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","age_right_clusters_correlation.png"))
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","age_right_clusters_correlation.pdf"))

plt.scatter(Xp[y==0, 2],age[y==0], c='b', label='off - resting state')
plt.scatter(Xp[y==1, 2], age[y==1],  c='r', label='off - resting state')
plt.xlabel("left cluster")
plt.ylabel("age")
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","age_left_clusters_correlation.png"))
plt.savefig(os.path.join(wm_folder,"pattern_interpretation","age_left_clusters_correlation.pdf"))
################################################################################


#plot with the weight map associated to eaach fold (leave one out, so 37 weights maps)
################################################################################
DATA_X = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/T.npy"
DATA_y = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/y_state.npy"
CONFIG = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/config_dCV.json"
RESULTS_CSV = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/results_dCV_recall_mean.xlsx"
RESULTS_PATH = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/model_selectionCV"
number_subjects = 37

OUTPUT = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/weights_maps"
results = pd.read_excel(RESULTS_CSV,sheetname = 2)
Xtot = np.load(DATA_X)
number_samples = Xtot.shape[0]
vmax = 0.001
thresh_norm_ratio = 0.99
thresh_size = 10


babel_mask  = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

weight_maps = np.zeros((number_subjects,number_features))

config_file = open(CONFIG)
config = json.load(config_file)
# Extract a single score for each cluster
K = 2 # nb cluster of interest in only 2: left and right
scores = np.zeros((number_samples, 2))

for cv  in range(number_subjects):
    if cv <10:
        cv_pattern  = "cv0"+str(cv)        
    else:
        cv_pattern  = "cv"+str(cv)
    path = os.path.join(RESULTS_PATH,cv_pattern)        
    print (path)
    argmax = results[results.index == cv].param_key.item()
    print (argmax)
    path_argmax = os.path.join(path,"refit",argmax)
    
    train_labels = config["resample"][cv_pattern+"/refit"][0]
    test_labels = config["resample"][cv_pattern+"/refit"][1]
    X_test = Xtot[test_labels,:]
    
    # extract weight map
    beta = np.load(os.path.join(path_argmax,"beta.npz"))['arr_0']
    weight_maps[cv,:] = brainomics.array_utils.arr_threshold_from_norm2_ratio(beta, .99)[0][:,0]
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = beta.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    map_filename = os.path.join(path_argmax, "weight_map.nii.gz")
    out_im.to_filename(map_filename)

    beta = nibabel.load(map_filename).get_data()
    beta_t, t = brainomics.array_utils.arr_threshold_from_norm2_ratio(beta, .99)
   

#
#    fig = nilearn.plotting.plot_glass_brain(map_filename,colorbar=True,plot_abs=False,threshold = t, title = cv_pattern,vmax =0.001,vmin = -0.001)
#    fig.savefig(os.path.join(OUTPUT ,"weight_map_"+cv_pattern+".pdf"))
#    fig.savefig(os.path.join(OUTPUT ,"weight_map_"+cv_pattern+".png"))
#

    command = "/home/ad247405/git/scripts/brainomics/image_clusters_analysis_nilearn.py %s -o %s --vmax %f --thresh_norm_ratio %f --thresh_size %i" % \
   (map_filename, path_argmax, vmax, thresh_norm_ratio, thresh_size)  
    os.system(command)
    
    labels_img  = nibabel.load(os.path.join(path_argmax,"weight_map_clust_labels.nii.gz"))
    labels_arr = labels_img.get_data()
    labels_flt = labels_arr[mask_bool]
    mask_clusters = np.zeros((len(np.unique(labels_flt)),number_features))
    
    correlation_clusters = np.zeros((len(np.unique(labels_flt)),2))
    for k in np.unique(labels_flt):
        mask_clusters[k,:] = labels_flt == k
        print("Cluster:",k, "size:", mask_clusters[k,:].sum())
        
        correlation_clusters[k,0] = pearsonr(mask_clusters[k,:],mask_cluster_right)[0]
        correlation_clusters[k,1] = pearsonr(mask_clusters[k,:],mask_cluster_left)[0]
    mask_clusters = mask_clusters !=0 #convert in bool
    
    cluster_right_index = correlation_clusters[:,0].argmax()
    cluster_left_index = correlation_clusters[:,1].argmax()
    #assert cluster_right_index != cluster_left_index
    scores[test_labels, 0] = np.dot(X_test[:, mask_clusters[cluster_right_index,:]], beta[mask_clusters[cluster_right_index,:]]).ravel()
    scores[test_labels, 1] = np.dot(X_test[:, mask_clusters[cluster_left_index,:]], beta[mask_clusters[cluster_left_index,:]]).ravel()

    
    
#plot
################################################################################     
plt.scatter(scores[y==0, 0], scores[y==0, 1], c='b', label='off - resting state')
plt.scatter(scores[y==1, 0], scores[y==1, 1], c='r', label='transition toward hallucination')
plt.xlabel("right cluster")
plt.ylabel("left cluster")
plt.legend(loc = "upper left")
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","clusters_correlation.png"))
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","clusters_correlation.pdf"))


plt.scatter(scores[y==0, 0],age[y==0], c='b', label='off - resting state')
plt.scatter(scores[y==1, 0], age[y==1],  c='r', label='off - resting state')
plt.xlabel("right cluster")
plt.ylabel("age")
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","age_right_clusters_correlation.png"))
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","age_right_clusters_correlation.pdf"))

plt.scatter(scores[y==0, 1],age[y==0], c='b', label='off - resting state')
plt.scatter(scores[y==1, 1], age[y==1],  c='r', label='off - resting state')
plt.xlabel("left cluster")
plt.ylabel("age")
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","age_left_clusters_correlation.png"))
plt.savefig(os.path.join("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/pattern_interpretation","age_left_clusters_correlation.pdf"))





#compute dice index


dice_bar(weight_maps.T)[0]  #0.78
dice_bar(weight_maps_svm.T)[0]  #0.84


def dice_bar(thresh_comp):
    """Given an array of thresholded component of size n_voxels x n_folds,
    compute the average DICE coefficient.
    """
    n_voxels, n_folds = thresh_comp.shape
    # Paire-wise DICE coefficient (there is the same number than
    # pair-wise correlations)
    n_corr = int(n_folds * (n_folds - 1) / 2)
    thresh_comp_n0 = thresh_comp != 0
    # Index of lines (folds) to use
    ij = [[i, j] for i in range(n_folds) for j in range(i + 1, n_folds)]
    num =([2 * (np.sum(thresh_comp_n0[:,idx[0]] & thresh_comp_n0[:,idx[1]]))
    for idx in ij])

    denom = [(np.sum(thresh_comp_n0[:,idx[0]]) + \
              np.sum(thresh_comp_n0[:,idx[1]]))
             for idx in ij]
    dices = np.array([float(num[i]) / denom[i] for i in range(n_corr)])
 
    return dices.mean(), dices
  


# SVM model selection
RESULTS_CSV_SVM = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/results_dCV_recall_mean.xlsx"
results = pd.read_excel(RESULTS_CSV_SVM,sheetname = 2)
RESULTS_PATH = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/model_selectionCV"
number_subjects = 37
weight_maps_svm = np.zeros((number_subjects,number_features))
OUTPUT_SVM = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/weights_maps"
for cv  in range(number_subjects):
    if cv <10:
        cv_pattern  = "cv0"+str(cv)        
    else:
        cv_pattern  = "cv"+str(cv)
    path = os.path.join(RESULTS_PATH,cv_pattern)        
    print (path)
    argmax = results[results.index == cv].param_key.item()
    print (argmax)
    if argmax == 10.0:
        argmax = 10
    path_argmax = os.path.join(path,"refit",str(argmax))
    
    # extract weight map
    beta = np.load(os.path.join(path_argmax,"beta.npz"))['arr_0'].T
    weight_maps_svm[cv,:] = brainomics.array_utils.arr_threshold_from_norm2_ratio(beta, .99)[0][:,0]

    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = beta.ravel()
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    map_filename = os.path.join(path_argmax, "weight_map.nii.gz")
    out_im.to_filename(map_filename)

    beta = nibabel.load(map_filename).get_data()
    beta_t, t = brainomics.array_utils.arr_threshold_from_norm2_ratio(beta, .99)
 


    fig = nilearn.plotting.plot_glass_brain(map_filename,colorbar=True,plot_abs=False,threshold = t, title = cv_pattern)
    fig.savefig(os.path.join(OUTPUT_SVM ,"weight_map_"+cv_pattern+".pdf"))
    fig.savefig(os.path.join(OUTPUT_SVM ,"weight_map_"+cv_pattern+".png"))
