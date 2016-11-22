
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
#INPUT_CSV = os.path.join(BASE_PATH,"population.txt")
INPUT_CSV = os.path.join(BASE_PATH,"population26oct.txt")
##################################################################################
#mask_bool = nibabel.load( '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/mask.nii.gz').get_data()
#mask_bool= np.array(mask_bool !=0)
#number_features = mask_bool.sum()
#############################################################################
mask_bool = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz').get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

#############################################################################
#First Level analysis, compute one contrast T map per subject
block_index = 0
pop = pd.read_csv(INPUT_CSV,delimiter=' ')
#pop.drop(pop.index[16])
number_subjects = pop.shape[0]
img_shape = (53, 63, 52)
T = []
betas = []
y_state = []
subject = []

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
                y_state.append(1)
                subject.append(i)
                period_number= period_number +1
        if subject_labels[j] =='OFF':
            state_of_interest_bool[j] = True
            labels_periods[index] = period_number
            labels_on_trans[index] = 0
            index= index +1
            if subject_labels[j+1] != 'OFF':
                y_state.append(0)
                subject.append(i)
                period_number= period_number +1
            
            
    #Keep only Trans and off volumes for further analysis        
    X=X[state_of_interest_bool==1,:]
       
    #Create Design Matrix with ramps
    design_mat = np.zeros((sum(subject_labels=='TRANS') + sum(subject_labels=='OFF'), period_number - 1))
    for k in range(1,period_number):
        design_mat[labels_periods == k,k-1]= np.arange(1,sum(labels_periods==k)+1)
     
     
    contrasts=np.identity((period_number -1))
     
    #Remove inter subject variability        
    mean_X=( X[labels_on_trans ==0].mean(axis=0) +  X[labels_on_trans ==1].mean(axis=0) ) / float(2)      
    X=X-mean_X       
                      
    #Univariate statistics    
    muols = MUOLS(Y=X,X=design_mat)
    muols.fit()
    tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
    for m in range (0,period_number-1):
            T.append(tvals[m])
            betas.append(tvals[m])
            block_index = block_index+1
            print block_index

    print i

T = np.array(T)
betas = np.array(betas)
y_state = np.array(y_state)
subject = np.array(subject)
##################################

T = np.nan_to_num(T)


 #Store variables
##############################################################################
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','T.npy'),T)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','betas.npy'),betas)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','y_state.npy'),y_state)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','subject.npy'),subject)


#np.save(os.path.join(BASE_PATH,'results','multivariate_analysis','without_subject19','data_without_subject19','T.npy'),T)
#np.save(os.path.join(BASE_PATH,'results','multivariate_analysis','without_subject19','data_without_subject19','betas.npy'),betas)
#np.save(os.path.join(BASE_PATH,'results','multivariate_analysis','without_subject19','data_without_subject19','y_state.npy'),y_state)
#np.save(os.path.join(BASE_PATH,'results','multivariate_analysis','without_subject19','data_without_subject19','subject.npy'),subject)
