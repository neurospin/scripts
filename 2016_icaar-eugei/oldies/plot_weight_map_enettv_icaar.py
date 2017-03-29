# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:59:10 2016

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

BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR'
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/model_selection'
POPULATION_CSV = os.path.join(BASE_PATH,"population.csv")
INPUT_XL = os.path.join(WD,'results_dCV.xlsx')


##################################################################################
babel_mask  = nibabel.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################

pop = pd.read_csv(POPULATION_CSV,delimiter=' ')
number_subjects = pop.shape[0]

folds_summary = pd.read_excel(INPUT_XL,sheetname =2 )
path_model = os.path.join(WD,"model_selectionCV")
N_FOLDS = 5
penalty_start = 3
    
beta_all_folds = np.zeros((N_FOLDS,number_features))
for i in range(N_FOLDS):
    path_fold = os.path.join(path_model,'cv%02d') %i    
    best_param = folds_summary.param_key.ix[0]   
    beta_path = os.path.join(path_fold,'refit',str(best_param),'beta.npz')
    beta_all_folds[i,:] = np.load(beta_path)['arr_0'][ penalty_start:].reshape(number_features)

    

mean_beta = beta_all_folds.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_beta



out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
  
  
  
import nilearn  
from nilearn import plotting
from nilearn import image
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)







b0 = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection/results/0/0.1_0.4_0.4_0.2/beta.npz')['arr_0']
b1 = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection/results/1/0.1_0.4_0.4_0.2/beta.npz')['arr_0']
b2 = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection/results/2/0.1_0.4_0.4_0.2/beta.npz')['arr_0']
b3 = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection/results/3/0.1_0.4_0.4_0.2/beta.npz')['arr_0']
b4 = np.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection/results/4/0.1_0.4_0.4_0.2/beta.npz')['arr_0']

beta= np.zeros((368123,5))
beta[:,0] = b0[:,0]
beta[:,1] = b1[:,0]
beta[:,2] = b2[:,0]
beta[:,3] = b3[:,0]
beta[:,4] = b4[:,0]

mean_beta = beta.mean(axis=1)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_beta[3:]
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/no_model_selection'
filename = os.path.join(WD,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)

