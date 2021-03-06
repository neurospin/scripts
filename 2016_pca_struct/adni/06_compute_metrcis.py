# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:59:39 2017

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
import scipy.stats
import array_utils
################
# Input/Output #
################
INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/adni/adni_model_selection_5x5folds"
INPUT_NEW_DIR = "/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"model_selectionCV")
INPUT_DIR_CORR = os.path.join(INPUT_NEW_DIR,"model_selectionCV_old")
INPUT_MASK_PATH = "/neurospin/brainomics/2016_pca_struct/adni/2017_GraphNet_adni_corrected_A_500ite_patients/mask.npy"
INPUT_RESULTS_FILE=os.path.join(INPUT_BASE_DIR,"results_dCV_5folds.xlsx")
INPUT_RESULTS_NEW_FILE=os.path.join(INPUT_NEW_DIR,"results_dCV_5folds.xlsx")

INPUT_CONFIG_FILE = os.path.join(INPUT_BASE_DIR,"config_dCV.json")

INPUT_DIR_GRAPHNET = "/neurospin/brainomics/2016_pca_struct/adni/2017_GraphNet_adni_corrected_A_500ite_patients"
INPUT_RESULTS_GRAPHNET = os.path.join(INPUT_DIR_GRAPHNET,"results_dCV_5folds.xlsx")
##############
# Parameters #
##############

N_COMP = 10
N_OUTER_FOLDS = 5
#number_features = int(nib.load(INPUT_MASK_PATH).get_data().sum())
# Load scores_cv and best params
####################################################################

frobenius_norm=np.zeros((4,N_OUTER_FOLDS))

frobenius_norm[0,:] = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/metrics_2017/frobenius_reconstruction_error_test_sparse.npy")[:,10]
frobenius_norm[1,:] = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/metrics_2017/frobenius_reconstruction_error_test_enet.npy")[:,10]
frobenius_norm[2,:] = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/metrics_2017/frobenius_reconstruction_error_test_tv.npy")[:,10]
frobenius_norm[3,:] = np.load("/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/metrics_2017/frobenius_reconstruction_error_test_gn.npy")[:,10]

print (frobenius_norm.mean(axis=1))

#Test Frobenius norm significance
tval, pval = scipy.stats.ttest_rel(frobenius_norm[0,:],frobenius_norm[2,:])
print(("Frobenius stats for TV vs sparse: T = %r , pvalue = %r ") %(tval, pval))
#Test Frobenius norm significance
tval, pval = scipy.stats.ttest_rel(frobenius_norm[1,:],frobenius_norm[2,:])
print(("Frobenius stats for TV vs Enet: T = %r , pvalue = %r ") %(tval, pval))


tval, pval = scipy.stats.ttest_rel(frobenius_norm[3,:],frobenius_norm[2,:])
print(("Frobenius stats for TV vs graphNet: T = %r , pvalue = %r ") %(tval, pval))

################################################################################
number_features = 317379
components_sparse = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
components_enet = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
components_tv = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))
components_gn = np.zeros((number_features,N_COMP,N_OUTER_FOLDS))

for cv in range(0,5):
    fold = "cv0%r" %(cv)
    components_sparse[:,:,cv] = np.load('/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV/'+fold+'/all/sparse_pca_0.0_0.0_1.0/components.npz')['arr_0']
    components_enet[:,:,cv] = np.load('/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV_old/'+fold+'/all/struct_pca_0.1_1e-06_0.1/components.npz')['arr_0']
    components_tv[:,:,cv] = np.load('/neurospin/brainomics/2016_pca_struct/adni/2017_adni_corrected/model_selectionCV_old/'+fold+'/all/struct_pca_0.1_0.5_0.1/components.npz')['arr_0']
    components_gn[:,:,cv] = np.load('/neurospin/brainomics/2016_pca_struct/adni/2017_GraphNet_adni_corrected_A_500ite_patients/model_selectionCV/'+fold+'/all/graphNet_pca_0.1_0.5_0.1/components.npz')['arr_0']


for i in range(10):
    for j in range(5):
        components_sparse[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_sparse[:,i,j], .99)
        components_enet[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_enet[:,i,j], .99)
        components_tv[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_tv[:,i,j], .99)
        components_gn[:,i,j],t = array_utils.arr_threshold_from_norm2_ratio(components_tv[:,i,j], .99)


components_tv = identify_comp(components_tv)
components_sparse = identify_comp(components_sparse)
components_enet = identify_comp(components_enet)
components_gn = identify_comp(components_gn)


def identify_comp(comp):
    for i in range(1,N_COMP):
        corr = np.zeros((10,5))
        for j in range(1,N_COMP):
            for k in range(1,N_OUTER_FOLDS):
                #corr[j,k] = np.abs(np.corrcoef(comp[:,i,0],comp[:,j,k]))[0,1]
                map = np.vstack((comp[:,i,0],comp[:,j,k]))
                corr[j,k] = dice_bar(map.T)[0]

        for k in range(1,N_OUTER_FOLDS):
            comp[:,i,k] = comp[:,np.argmax(corr,axis=0)[k],k]
    return comp



dice_sparse = list()
all_pairwise_dice_sparse = list()
dice_enet = list()
all_pairwise_dice_enet = list()
dice_enettv = list()
all_pairwise_dice_enettv = list()
dice_graphNet = list()
all_pairwise_dice_graphNet = list()
for i in range(N_COMP):
    dice_sparse.append(dice_bar(components_sparse[:,i,:])[0]) #mean of all 10 pairwise dice
    all_pairwise_dice_sparse.append(dice_bar(components_sparse[:,i,:])[1])
    dice_enet.append(dice_bar(components_enet[:,i,:])[0])
    all_pairwise_dice_enet.append(dice_bar(components_enet[:,i,:])[1])
    dice_enettv.append(dice_bar(components_tv[:,i,:])[0])
    all_pairwise_dice_enettv.append(dice_bar(components_tv[:,i,:])[1])
    dice_graphNet.append(dice_bar(components_gn[:,i,:])[0])
    all_pairwise_dice_graphNet.append(dice_bar(components_gn[:,i,:])[1])
print (np.mean(dice_sparse))
print (np.mean(dice_enet))
print (np.mean(dice_enettv))
print (np.mean(dice_graphNet))



sparse_plot= plt.plot(np.arange(1,11),dice_sparse,'b-o',markersize=5,label = "Sparse PCA")
enet_plot= plt.plot(np.arange(1,11),dice_enet,'g-^',markersize=5,label = "ElasticNet")
tv_plot= plt.plot(np.arange(1,11),dice_graphNet,'y-s',markersize=5,label = "PCA-GraphNet")
tv_plot= plt.plot(np.arange(1,11),dice_enettv,'r-s',markersize=5,label = "PCA-TV")
plt.xlabel("Components")
plt.ylabel("Dice index")
plt.legend(loc= 'upper right')
#plt.savefig('/neurospin/brainomics/2016_pca_struct/fmri/fmri_model_selection_5x5folds/metrics_2017/dice_index_per_component.pdf')


#########
#####################################################################
#Statistical test of Dice index
###############################################################################

# I want to test whether this list of pairwise diff is different from zero?
#(i.e TV leads to different results than sparse?).
#We cannot do a one-sample t-test since samples are not independant!
#Use of permuations
diff_sparse = np.mean(all_pairwise_dice_enettv,axis=0) - np.mean(all_pairwise_dice_sparse,axis=0)
diff_enet = np.mean(all_pairwise_dice_enettv,axis=0) - np.mean(all_pairwise_dice_enet,axis=0)
diff_gn = np.mean(all_pairwise_dice_enettv,axis=0) - np.mean(all_pairwise_dice_graphNet,axis=0)


pval = one_sample_permutation_test(y=diff_sparse,nperms = 1000)
print(("Dice index stats for TV vs sparse: pvalue = %r ") %(pval))
pval = one_sample_permutation_test(y=diff_enet,nperms = 1000)
print(("Dice index stats for TV vs Enet: pvalue = %r ") %(pval))
pval = one_sample_permutation_test(y=diff_gn,nperms = 1000)
print(("Dice index stats for TV vs GN: pvalue = %r ") %(pval))

###############################################################################





def one_sample_permutation_test(y,nperms):
    T,p =scipy.stats.ttest_1samp(y,0.0)
    max_t = list()

    for i in range(nperms):
            r=np.random.choice((-1,1),y.shape)
            y_p=r*abs(y)
            Tperm,pp =scipy.stats.ttest_1samp(y_p,0.0)
            Tperm= np.abs(Tperm)
            max_t.append(Tperm)
    max_t = np.array(max_t)
    pvalue = np.sum(max_t>np.abs(T)) / float(nperms)
    return pvalue


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
