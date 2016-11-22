# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:14:14 2016

@author: ad247405
"""


import nibabel as nibabel
import numpy as np
import os
import nilearn.signal
import nilearn.image
import re
import glob
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import random
import brainomics

# Read pop csv

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
INPUT_CSV = os.path.join(BASE_PATH,"population.txt")



##################################################################################
mask_nib = nibabel.load( '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/mask.nii.gz')
mask_bool = nibabel.load( '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/mask.nii.gz').get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
#############################################################################

mask_bool = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz').get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()





#############################################################################
#First Level analysis, compute one contrast T map per subject

pop = pd.read_csv(INPUT_CSV,delimiter=' ')
number_subjects = pop.shape[0]
img_shape = (53, 63, 52)
T=np.zeros((number_subjects,number_features))
for i in range(number_subjects):
    subject_data_path = pop.data_path.ix[i]
    X = nibabel.load(subject_data_path).get_data()    
    X = X[:,:,:,60:]
    Xr = np.zeros((X.shape[3],img_shape[0] * img_shape[1] * img_shape[2]))
    for k in range(X.shape[3]):
        Xr[k,:] = X[:,:,:,k].ravel()
    X = Xr

    X = X[:,mask_bool.ravel()] 

    #detrending   
    X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=0.01925)
    
    subject_labels = np.load(pop.state_path.ix[i])
     
    #Use labelisation infos
    labels_periods = np.zeros(sum(subject_labels=='TRANS')+sum(subject_labels=='OFF'))
    labels_on_trans = np.zeros(sum(subject_labels=='TRANS')+sum(subject_labels=='OFF')) 
    index = 0    
    period_number=1
    state_of_interest_bool = np.zeros(X.shape[0])
    for j in range(840):
        if subject_labels[j] =='TRANS':
            state_of_interest_bool[j] = True
            labels_periods[index] = period_number
            labels_on_trans[index] = 1
            index= index +1
            if subject_labels[j+1] != 'TRANS':
                period_number= period_number +1
        if subject_labels[j] =='OFF':
            state_of_interest_bool[j] = True
            labels_periods[index] = period_number
            labels_on_trans[index] = 0
            index= index +1
            if subject_labels[j+1] != 'OFF':
                period_number= period_number +1
            
            
    #Keep only Trans and off volumes for further analysis        
    X=X[state_of_interest_bool==1,:]
       
    #Create Design Matrix with ramps
    design_mat = np.zeros((sum(subject_labels=='TRANS') + sum(subject_labels=='OFF'), period_number - 1))
    contrast = np.zeros((period_number -1))
    for k in range(1,period_number):
        design_mat[labels_periods == k,k-1]= np.arange(1,sum(labels_periods==k)+1)
        if labels_on_trans[labels_periods==k][0] ==0:
            contrast[k-1] = -1
        if labels_on_trans[labels_periods==k][0] ==1:
            contrast[k-1] = 1  
            
            
         
    #Remove inter subject variability        
    mean_X=( X[labels_on_trans ==0].mean(axis=0) +  X[labels_on_trans ==1].mean(axis=0) ) / float(2)      
    X=X-mean_X       
                      
    #Univariate statistics    
    muols = MUOLS(Y=X,X=design_mat)
    muols.fit()
    tvals, pvals, dfs = muols.t_test(contrast,pval=True, two_tailed=True)
    T[i,:]=tvals
    print i


  
  


################################  
 
 #Store variables
#############################################################################
np.save(os.path.join(BASE_PATH,'results','univariate_analysis','T.npy'),T)
#############################################################################

#Retrieve variables
#############################################################################
T=np.load(os.path.join(BASE_PATH,'results','univariate_analysis','T.npy'))
#############################################################################
############################################################################


nib = nibabel.load(subject_data_path)
affine = nib.get_affine()

#Second Level analysis
############################################################################


design_mat=np.ones((number_subjects,1))
muols = MUOLS(Y=T,X=design_mat)
muols.fit()
contrast=np.zeros((1,1))
contrast[0]=1

#Uncorrected raw pvalue
tvals, pvals, dfs = muols.t_test(contrast,pval=True, two_tailed=True)

#Save tvals
arr = np.zeros(mask_bool.shape)
arr[mask_bool] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine = affine)
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis","tval_trans_off.nii.gz"))

#Save raw pvalue
arr = np.zeros(mask_bool.shape)
arr[mask_bool] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine = affine)
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis","pval_trans-off.nii.gz"))


#Save log raw pvalues
arr = np.zeros(mask_bool.shape)
log10_ps = -np.log10(pvals[0])
arr[mask_bool] = log10_ps
out_im = nibabel.Nifti1Image(arr,affine = affine)
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis","p-log10_trans-off.nii.gz"))



#Plot with glass brain
from nilearn import plotting
from nilearn import image
img_path = os.path.join(BASE_PATH,"results","univariate_analysis","p-log10_trans-off.nii.gz")

nilearn.plotting.plot_glass_brain(img_path,colorbar=True,threshold = 3.5,plot_abs=False,title = "univariate analysis: Transition toward hallucinations vs resting state")





#############################################################################
#############################################################################

#Corrected pvalue using Tmax with 1000 permutation


nperms=1000
two_tailed=True
DesignMat=np.ones((number_subjects-1,1))
muols = MUOLS(Y=T,X=DesignMat)
muols.fit()
contrast=np.zeros((1,1))
contrast[0]=1
tvals, pvals, dfs = muols.t_test(contrast,pval=True, two_tailed=True)
        
max_t = list()

for i in xrange(nperms): 
        r=np.zeros((number_subjects-1,1))
        r[:,0]=np.random.choice((-1,1),number_subjects-1)
        Tp=r*abs(T)

        muols = MUOLS(Y=Tp,X=DesignMat)
        muols.fit()
        tvals_perm,_,_ = muols.t_test(contrast,pval=False, two_tailed=True)
                               
        if two_tailed:
            tvals_perm = np.abs(tvals_perm)
        max_t.append(np.nanmax(tvals_perm, axis=1))
        del muols
        print i
            
max_t = np.array(max_t)
tvals_ = np.abs(tvals) if two_tailed else tvals
pvalues = np.array([np.array([np.sum(max_t[:, con] >= t) for t in tvals_[con, :]])\
                / float(nperms) for con in xrange(contrast.shape[0])])

pvalues = np.nan_to_num(pvalues)
#############################################################################
#############################################################################

#Save corrected pvalues
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = pvalues[0]
out_im = nibabel.Nifti1Image(arr, affine=affine)
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis","corrected_pval_perm_trans_off.nii.gz"))

#Save log10corrected pvalues
log10_ps = -np.log10(pvalues[0])
arr = np.zeros(mask_bool.shape)
arr[mask_bool] = log10_ps
out_im = nibabel.Nifti1Image(arr, affine=affine)
out_im.to_filename(os.path.join(BASE_PATH,"results","univariate_analysis","corrected_p-log10_trans_off.nii.gz"))

img_path = os.path.join(BASE_PATH,"results","univariate_analysis","corrected_pval_perm_trans_off.nii.gz")
nilearn.plotting.plot_glass_brain(img_path,plot_abs=False,title = "univariate analysis: Transition toward hallucinations vs resting state")







#
#
#
#
#
##Mask implicit
###############################################################################
#mask_bool = nibabel.load( '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/mask.nii').get_data()
#mask_bool= np.array(mask_bool !=0)
#arr = np.zeros(mask_bool.shape)
#imp = np.zeros((35,53,63,52))
#for i in range(0,35):
#    arr[mask_bool] =T[i,:]
#    imp[i,:,:,:] = (np.isnan(arr) == False)
#
#for i in range(0,35):
#    true_mask = true_mask & imp[i,:].astype(bool)       
#
#out_im = nibabel.Nifti1Image(true_mask.astype(int), affine = affine)
#out_im.to_filename(os.path.join(BASE_PATH,"results","mask.nii.gz"))