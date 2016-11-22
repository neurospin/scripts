# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:57:49 2016

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

BASE_PATH = BASE_PATH= '/neurospin/brainomics/2016_AUSZ/results/VBM'


##################################################################################
babel_mask  = nibabel.load('/neurospin/brainomics/2016_AUSZ/results/VBM/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################
#cONTROLS VS SCZ
beta_path = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_controls/results/0/0.1_0.2_0.2_0.6/beta.npz'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_controls/results/0/0.1_0.2_0.2_0.6'

# SCZ VS ASD
beta_path ='/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_asd/results/0/0.1_0.6_0.0_0.4/beta.npz'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_asd/results/0/0.1_0.6_0.0_0.4'

# SCZ VS scz-ASD
beta_path = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_scz-asd/results/0/0.1_0.8_0.0_0.2/beta.npz'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_vs_scz-asd/results/0/0.1_0.8_0.0_0.2'


# SCZ VS scz-ASD
beta_path = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_asd_vs_controls/results/0/0.1_0.06_0.54_0.4/beta.npz'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/scz_asd_vs_controls/results/0/0.1_0.06_0.54_0.4'

# CONTROLS VS ASD
beta_path = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/asd_vs_controls/results/0/0.1_0.2_0.2_0.6/beta.npz'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/no_model_selection/asd_vs_controls/results/0/0.1_0.2_0.2_0.6'



beta = np.load(beta_path)['arr_0']
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta[penalty_start:,0]
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())

filename = os.path.join(WD,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
  
  
  
import nilearn  
from nilearn import plotting
from nilearn import image
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)


