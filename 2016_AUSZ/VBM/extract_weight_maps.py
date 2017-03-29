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
import nilearn  
from nilearn import plotting
from nilearn import image
import array_utils

BASE_PATH= '/neurospin/brainomics/2016_AUSZ/results/VBM'


##################################################################################
babel_mask  = nibabel.load('/neurospin/brainomics/2016_AUSZ/results/VBM/data/mask.nii')
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################



#SVM
WD = "/neurospin/brainomics/2016_AUSZ/results/VBM/linear_regression/model_selectionCV/\
refit/refit/0.1_0.35_0.35_0.3"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)





















#CONTROLS VS SCZ
dir_path = os.path.join(BASE_PATH,"scz_vs_controls")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_scz_vs_controls","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0]
scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.01]



best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_scz_vs_controls","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t,vmin=-0.002,vmax=0.002)




# SCZ VS ASD
dir_path = os.path.join(BASE_PATH,"scz_vs_asd")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_scz_vs_asd","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0.1]
scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.1]
scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean<0.6]


best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_scz_vs_asd","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)




# SCZ VS scz-ASD
dir_path = os.path.join(BASE_PATH,"scz_vs_scz-asd")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_scz_vs_scz-asd","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean<0.6]


best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_scz_vs_scz-asd","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)





# controls VS scz-ASD
dir_path = os.path.join(BASE_PATH,"scz_asd_vs_controls")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_scz_asd_vs_controls","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean<0.6]


best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_scz_asd_vs_controls","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)



# scz_asd_vs_asd
dir_path = os.path.join(BASE_PATH,"scz_asd_vs_asd")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_scz_asd_vs_asd","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0]
scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean<0.6]


best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_scz_asd_vs_asd","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)




# scz_asd_vs_asd
dir_path = os.path.join(BASE_PATH,"asd_vs_controls")
scores_all_svm = pd.read_excel(os.path.join(dir_path,"svm","results_dCV.xlsx"),sheetname = 0)
scores_all_enettv = pd.read_excel(os.path.join(dir_path,"enettv_asd_vs_controls","results_dCV.xlsx"),sheetname = 0)

scores_all_enettv = scores_all_enettv[scores_all_enettv.tv>0]
scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean>0.1]
#scores_all_enettv = scores_all_enettv[scores_all_enettv.prop_non_zeros_mean<0.6]


best_svm_param = str(scores_all_svm["param_key"][scores_all_svm["recall_mean"].argmax()])
best_enettv_param = scores_all_enettv["param_key"][scores_all_enettv["recall_mean"].argmax()]

WD_svm = os.path.join(dir_path,"svm","results","all","all",best_svm_param)
WD_enettv = os.path.join(dir_path,"enettv_asd_vs_controls","results","all","all",best_enettv_param)
                                  
beta = np.load(os.path.join(WD_svm,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_svm,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

beta = np.load(os.path.join(WD_enettv,"beta.npz"))['arr_0'][penalty_start:,0]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD_enettv,"mean_predictive_map_across_folds.nii.gz")
out_im.to_filename(filename)
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
#play with parameter vmax and vmin
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

#########################################################################################

