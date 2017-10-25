#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 10:47:10 2017

@author: ad247405
"""

import os
import numpy as np
import nibabel
import array_utils
import nilearn
from nilearn import plotting
from nilearn import image

BASE_PATH="/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/"
POPULATION_CSV = os.path.join(BASE_PATH,"population.csv")
MASK_PATH = os.path.join(BASE_PATH,"data","mask.nii")


##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
##################################################################################
penalty_start = 4

#
#SVM
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/svm/svm_schizCo+VIP_all/model_selectionCV/all/all/0.0001"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][0,penalty_start:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",cut_coords=(-1,-13,14),vmax=0.0005)

#Enet-
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enet_0.01_0.1_0"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",cut_coords=(-1,-13,14),vmax=0.1)

#Enet-GN
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enetgn_0.01_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",cut_coords=(-1,-13,14),vmax=0.01)



#Enet-TV
#WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
#results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.01_0.1_0.8"

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",vmax = 0.005)
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",vmax = 0.0005,cut_coords=(-1,-13,14))




#PLOT EACH CLUSTER FOR VIZU

#1 cingulate gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",vmax = 0.002,cut_coords=(3,-31,25))
#2 Right caudate
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",vmax = 0.001,cut_coords=(15,6,15))


#3 Precentral gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
                               vmax = 0.01,cut_coords=(-33,-21,63))

#4 Cingulate Gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
                               vmax = 0.001,cut_coords=(-3,30,18))


#5 Temporal pole
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax = 0.001,cut_coords=(21,1,-36))

#6 Left hippcampus and amygdala
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
                               vmax = 0.0005,cut_coords=(-25,-10,-25))



#7 Left caudate
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold =  0.000017,\
                               vmax = 0.0005,cut_coords=(-17,6,-1))

#8 Left thalamus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold =  0.000017,\
                               vmax = 0.001,cut_coords=(-7,-19,7))


#9 Right thalamus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax = 0.001,cut_coords=(6,-7,13))


#10 Middle temporal gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax = 0.0005,cut_coords=(-66,-37,-15))
