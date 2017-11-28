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
import matplotlib.pyplot as plt


MASK_PATH = "/neurospin/brainomics/2016_AUSZ/september_2017/results/\
VBM/data/data_with_intercept/mask.nii"


##################################################################################
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
penalty_start = 3
##################################################################################

#
#Ridge regression

WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/5cv/refit/refit/Ridge_1.0_0.9_0"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
#                               cut_coords=(49,-39,23),vmax=0.0005)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(8,-31,14))

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/weights_maps/ridge_regression.png")

#
#Enet regression

WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/5cv/refit/refit/enet_0.1_0.1_0"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()

#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
#                               cut_coords=(49,-39,23))
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(8,-31,14))

plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/weights_maps/enet_regression.png")


#Enettv
#WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
#linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/5cv/refit/refit/enettv_0.01_0.1_0.8"
#
#WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
#linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/5cv/refit/refit/enettv_1.0_0.9_0.6"
#
WD = "/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/5cv/refit/refit/enettv_1.0_0.9_0.5"

beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][penalty_start:,:]
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()
beta_t,t = array_utils.arr_threshold_from_norm2_ratio(beta, .99)

#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
#                               cut_coords=(49,-39,23),vmax=0.008)

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=False,threshold = "auto",\
                               cut_coords=(8,-31,14),vmax=0.008)



plt.savefig("/neurospin/brainomics/2016_AUSZ/september_2017/results/VBM/\
linear_regression_TV_ASD_SCZ_MAASCtot_intercept_penalty_start/weights_maps/enettv_regression.png")



#enettv_1.0_0.9_0.5
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(9,-82,4))


nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(43,-81,1))

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(46,-56,13))


nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(48,-36,-25))

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(57,-42,24))

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(-10,-40,75))

nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
                               vmax=0.008,cut_coords=(-52,-27,-6))

#enettv_0.01_0.1_0.8
#######################################

#Precuneus cortex
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(49,-49,-22))



# Occipital cortex
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(43,-84,3))


# Temporal gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(-53,-31,-1))

# Postcentral gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(50,-24,46))


# ANterior cingualte gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(3,21,22))

#Precuneus cortex
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = t,\
                               vmax=0.005,cut_coords=(-3,-49,15))


#
##PLOT EACH CLUSTER FOR VIZU
#
#1 cingulate gyrus
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",vmax = 0.002,cut_coords=(3,-31,25))
#2 Right caudate
nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",vmax = 0.001,cut_coords=(15,6,15))

#
##3 Precentral gyrus
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
#                               vmax = 0.01,cut_coords=(-33,-21,63))
#
##4 Cingulate Gyrus
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
#                               vmax = 0.001,cut_coords=(-3,30,18))
#
#
##5 Temporal pole
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
#                               vmax = 0.001,cut_coords=(21,1,-36))
#
##6 Left hippcampus and amygdala
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = 0.000017,\
#                               vmax = 0.0005,cut_coords=(-25,-10,-25))
#
#
#
##7 Left caudate
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold =  0.000017,\
#                               vmax = 0.0005,cut_coords=(-17,6,-1))
#
##8 Left thalamus
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold =  0.000017,\
#                               vmax = 0.001,cut_coords=(-7,-19,7))
#
#
##9 Right thalamus
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
#                               vmax = 0.001,cut_coords=(6,-7,13))
#
#
##10 Middle temporal gyrus
#nilearn.plotting.plot_stat_map(filename,colorbar=True,draw_cross=True,threshold = "auto",\
#                               vmax = 0.0005,cut_coords=(-66,-37,-15))
