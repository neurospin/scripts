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
#Enet-TV
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/0.1_0.04_0.36_0.6'

#Enet-tv for IMA
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_IMA_model_selection/0.1_0.0_0.6_0.4/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_IMA_model_selection/0.1_0.0_0.6_0.4'

#Enet-tv for RS
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_RS_model_selection/0.1_0.0_0.6_0.4/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_RS_model_selection/0.1_0.0_0.6_0.4'


#svm 
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/10/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/10'

#svm for IMA
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_IMA_model_selection/1e-05/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_IMA_model_selection/1e-05'


#svm for RS
beta = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_RS_model_selection/1e-05/beta.npz')['arr_0']
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_RS_model_selection/1e-05'



arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
  
  
  
  
  
  
beta = nibabel.load(filename).get_data()
import array_utils
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
  
import nilearn  
from nilearn import plotting
from nilearn import image
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t, vmax =0.001,vmin = -0.001)



