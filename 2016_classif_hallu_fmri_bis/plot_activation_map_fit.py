#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:40:10 2017

@author: ad247405
"""
import nibabel 
import numpy as np
import nilearn
import matplotlib.pyplot as plt


WD = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/vizu"
T = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/T_hallu_only.npy")
beta =  T[0,:]
mask_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/data/MNI152_T1_3mm_brain_mask.nii.gz"   
mask = nibabel.load(mask_path).get_data()
mask_bool = mask_bool= np.array(mask !=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = beta.ravel()
out_im = nibabel.Nifti1Image(arr, affine=nib.load(mask_path).get_affine())
filename = os.path.join(WD,"weight_map.nii.gz")
out_im.to_filename(filename)
beta = nibabel.load(filename).get_data()
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/vizu/activation_map.pdf")
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/vizu/activation_map.png")


#subject = 0
x = np.arange(8)
y=X[7:15,T[1,:].argmax()]
m,b = np.polyfit(x,y,1)


x = np.arange(8)
y=X[7:15,T[1,:].argmin()]
m,b = np.polyfit(x,y,1)


plt.plot(x, y, 'o',label = "samples")
plt.plot(x, m*x + b, '-',label = "fit")
plt.xlabel("Time point")
plt.ylabel("BOLD fMRI signal")
plt.legend(loc = "upper left")


plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/vizu/fit.pdf")
plt.savefig("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/vizu/fit.png")
