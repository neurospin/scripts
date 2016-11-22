# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:01:17 2016

@author: ad247405
"""

import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import nilearn.image
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn import grid_search, metrics
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import scipy
from scipy import stats 
from sklearn.preprocessing import StandardScaler      
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS    
import csv
from sklearn.metrics import roc_auc_score, recall_score
 
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
INPUT_CSV = os.path.join(BASE_PATH,"population.txt")
INPUT_SUBJECTS_LIST = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/data/Datas_sujets_sains/list_healthy_subjects_resting_state.txt'


##################################################################################
mask_bool = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz').get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

#############################################################################
     
#Compute Tstats image for each block 
#############################################################################

T = []
betas = []
y_state = []
subject = []
t=0
subject_num =0

DesignMat = np.zeros((900,8)) 
DesignMat[100:107, 0] = np.arange(1, 8)
DesignMat[200:207, 1] = np.arange(1, 8)
DesignMat[300:307, 2] = np.arange(1, 8)
DesignMat[400:407, 3] = np.arange(1, 8)
DesignMat[500:507, 4] = np.arange(1, 8)
DesignMat[600:607, 5] = np.arange(1, 8)
DesignMat[700:707, 6] = np.arange(1, 8)
DesignMat[800:807, 7] = np.arange(1, 8)
  
               
inf=open(INPUT_SUBJECTS_LIST, "r")                                  
                
for subject_folder in inf.readlines():
    subject_folder = subject_folder.replace("\n","") 
    subject_name = os.path.basename(subject_folder)
    print subject_name
    path_presto = glob.glob(subject_folder+"/w*")[0]
    subject_num=subject_num + 1
    babel_image = nibabel.load(path_presto)
    X = babel_image.get_data()
    Xr = np.zeros((X.shape[3],X.shape[0] * X.shape[1] * X.shape[2]))
    for k in range(X.shape[3]):
        Xr[k,:] = X[:,:,:,k].ravel()
    X = Xr
    X = X[:,mask_bool.ravel()] 

    #detrending   
    X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=0.01925)
              
          
         
    muols = MUOLS(Y=X,X=DesignMat)
    muols.fit()
    contrasts=np.identity((8))
    tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
    for m in range (0,8):
         T.append(tvals[m])
         betas.append(muols.coef[m,:])
         subject.append(subject_num)
         t=t+1
         print t
                 


T = np.array(T)
betas = np.array(betas)
y_state = np.array(y_state)
subject = np.array(subject)
##################################

T = np.nan_to_num(T)

#Conserve only block from 16 subjects to be coherent with IMA samples

T = T[:128,:]
betas = betas[:128,:]
subject = subject[:128]

          
y=np.zeros((128))     
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"           
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','T_RS.npy'),T)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','betas_RS.npy'),betas)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','y_RS.npy'),y)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','subject_RS.npy'),subject) 
                   


