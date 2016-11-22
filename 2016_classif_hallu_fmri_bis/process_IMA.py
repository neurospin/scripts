# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 10:47:35 2016

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:29:08 2015

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


##################################################################################
mask_bool = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz').get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

#############################################################################
     
#Compute Tstats image for each block 
#############################################################################

T=np.zeros((128,number_features))
betas=np.zeros((128,number_features))

subject=np.zeros((128))  
subject_num=0
t=0

from nipy.modalities.fmri import hemodynamic_models 
hrf=hemodynamic_models.spm_hrf(tr=3.0,oversampling=1,onset=0.0,time_length=400)

DesignMat = np.zeros((132,8)) 
DesignMat[20:27:, 0] = np.arange(1, 8)
DesignMat[34:41:, 1] = np.arange(1, 8)
DesignMat[48:55, 2] = np.arange(1, 8)
DesignMat[62:69, 3] = np.arange(1, 8)
DesignMat[76:83, 4] = np.arange(1, 8)
DesignMat[90:97, 5] = np.arange(1, 8)
DesignMat[104:111, 6] = np.arange(1, 8)
DesignMat[118:125, 7] = np.arange(1, 8)
  

lab = np.zeros((132)) 
lab[20:27] = 1
lab[34:41] = 1
lab[48:55] = 1
lab[62:69] = 1
lab[76:83] = 1
lab[90:97] = 1
lab[104:111] = 1
lab[118:125] = 1  
#Convolve regressor with HRF          
for i in range(0,8):
    DesignMat[:,i] = np.convolve(DesignMat[:,i],hrf)[0:132]
                   
               
BASE_PATH ="/neurospin/brainomics/2016_classif_hallu_fmri/data/DATA_Localizer/Sujets_sains_proc"
                                   
                
for i in range(1,20):
    imagefile_pattern = 'Sujet'+str(i)+'_'
    for file in os.listdir(BASE_PATH):
        if re.match(imagefile_pattern, file):
            name= file
            print(name)
            path=os.path.join(BASE_PATH,name,'SPM')
            subject_num=subject_num + 1
            images = list()
            pathlist=list()
            all_scans=sorted(glob.glob(os.path.join(path,'wraf*')))
           
            for imagefile_name in all_scans:
               pathlist.append(imagefile_name)
               babel_image = nibabel.load(imagefile_name)
               #babel_image=nilearn.image.smooth_img(imgs=babel_image, fwhm=6)
              # babel_image=nilearn.image.resample_img(babel_image, target_affine=babel_image.get_affine()*2, target_shape=mask_bool.shape, interpolation='continuous', copy=True, order='F')
               images.append(babel_image.get_data().ravel())
            
            X = np.vstack(images)
            X = X[:, mask_bool.ravel()] 
            X=nilearn.signal.clean(X,detrend=True,standardize=True,confounds=None,low_pass=None, high_pass=None, t_r=3)
            
          
#            X=X-X.mean(axis=0)             
#            
            muols = MUOLS(Y=X,X=DesignMat)
            muols.fit()
            contrasts=np.identity((8))
            tvals, pvals, dfs = muols.t_test(contrasts,pval=True, two_tailed=True)
            for m in range (0,8):
                 T[t,:]=tvals[m]
                 betas[t,:]=muols.coef[m,:]
                 subject[t]=subject_num
                 t=t+1
                 print t
                 

T = np.array(T)
betas = np.array(betas)
subject = np.array(subject)
          
          
          
y=np.zeros((128))     
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"           
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','T_IMA.npy'),T)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','betas_IMA.npy'),betas)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','y_IMA.npy'),y)
np.save(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','subject_IMA.npy'),subject) 
                   



