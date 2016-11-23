# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:53:58 2016

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
  
import nilearn  
from nilearn import plotting
from nilearn import image
import array_utils

BASE_PATH="/neurospin/brainomics/2016_icaar-eugei"

##################################################################################

#ICAAR
babel_mask  = nibabel.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
#
#Enet-TV
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/enettv/model_selection/0.1_0.4_0.4_0.2'
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]

arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)

beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmax =0.005,vmin = -0.005)
 
 
 #svm f
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR/svm/model_selection/1e-06'
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)

beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

 
  ##################################################################################
#ICAAR+EUGEI
babel_mask  = nibabel.load('/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()

#Enet-TV
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/enettv/model_selection/0.1_0.18_0.02_0.8'
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]

arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)

beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmax =0.001,vmin = -0.001)
 
 
 #svm f
WD = '/neurospin/brainomics/2016_icaar-eugei/results/VBM/ICAAR+EUGEI/svm/model_selection/1e-05'
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)

beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
