# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:18:09 2016

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:49:50 2016

@author: ad247405
"""
import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from collections import OrderedDict

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
POPULATION_CSV = os.path.join(BASE_PATH,"population.txt")
INPUT_XL = os.path.join(WD,'results_dCV.xlsx')


##################################################################################
babel_mask  = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
#
#pop = pd.read_csv(POPULATION_CSV,delimiter=' ')
#number_subjects = pop.shape[0]
#
#folds_summary = pd.read_excel(INPUT_XL,sheetname =2 )
#path_model = os.path.join(WD,"model_selectionCV")
#
#beta_all_folds = np.zeros((number_subjects,number_features))
#for i in range(number_subjects):
#    path_fold = os.path.join(path_model,'cv%02d') %i    
#    best_param = folds_summary.param_key.ix[0]   
#    beta_path = os.path.join(path_fold,'refit',str(best_param),'beta.npz')
#    beta_all_folds[i,:] = np.load(beta_path)['arr_0'].reshape(number_features)

#mean_beta = beta_all_folds.mean(axis=0)
#arr = np.zeros(mask_bool.shape);
#arr[mask_bool] = mean_beta

#
#Enet-TV
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6'

#Enet-tv for IMA
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_IMA_model_selection/0.1_0.0_0.6_0.4/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_IMA_model_selection/0.1_0.0_0.6_0.4'

#Enet-tv for RS
beta = np.load('')['arr_0']
WD = ''


arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map_0.1_0.0_0.6_0.4.nii.gz")
out_im.to_filename(filename)
  
  
  
  
  
  
beta = nibabel.load(filename).get_data()
import array_utils
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
  
import nilearn  
from nilearn import plotting
from nilearn import image
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

