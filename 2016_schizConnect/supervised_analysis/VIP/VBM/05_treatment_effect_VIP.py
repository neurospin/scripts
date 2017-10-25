

"""
Created on Fri Oct 13 09:21:46 CEST 2017

@author: amicie.depierrefeu@cea.fr
"""



import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil
import mulm
from mulm import MUOLS
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
from nilearn import image

INPUT_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mean_centered_by_site_all/X.npy"
INPUT_Y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mean_centered_by_site_all/y.npy"
INPUT_DOSE_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/dose_ongoing_treatment.npy"
INPUT_MASK = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mean_centered_by_site_all/mask.nii"
SITE = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy"

OUTPUT = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/treatment_effect"
#############################################################################
penalty_start = 2

X = np.load(INPUT_X)
y = np.load(INPUT_Y)
site = np.load(SITE)

X = X[site==4,:]
y = y[site==4]
assert X.shape[0] ==  y.shape[0] == 92

X = X[(y==1).ravel(),:]
dose = np.load(INPUT_DOSE_VIP)
assert X.shape[0] == dose.shape[0] == 39

# MULM
mask_ima = nibabel.load(INPUT_MASK)
mask_arr = mask_ima.get_data() != 0


Z = X[:, :penalty_start]
Y = X[: , penalty_start:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # dose, age, sex
DesignMat[:, 0] = dose.ravel() # dose
DesignMat[:, 1] = Z[:, 0]  # age
DesignMat[:, 2] = Z[:, 1]  # sex

muols = MUOLS(Y=Y,X=DesignMat)
muols.fit()
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0], pval=True)

plt.hist(pvals.T)
plt.ylabel("count")
plt.xlabel("pvalues")

plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/results/treatment_effect/distribution_pvalues.png")

arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
filemane = out_im
out_im.to_filename(os.path.join(OUTPUT ,"t_stat_treatment_effect.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_treatment_effect.nii.gz"))


arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_treatment_effect.nii.gz"))



#Nilearn vizualisation of tstap map to include in supplementary materials
#1) T stat map
plt.figure()
nilearn.plotting.plot_stat_map(os.path.join(OUTPUT ,"t_stat_treatment_effect.nii.gz"),\
                               colorbar=True,draw_cross=False,threshold = "auto",cut_coords=(-1,-13,14),vmax=3)

plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/treatment_T_stat_map.png")

#2) pval <0.05 map
plt.figure()
nilearn.plotting.plot_stat_map(os.path.join(OUTPUT ,"log10pval_treatment_effect.nii.gz"),\
                               colorbar=True,draw_cross=False,threshold = 1.3,cut_coords=(-1,-13,14))

plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/treatment_pval0.05_map.png")


#3) pval <0.05 map
plt.figure()
nilearn.plotting.plot_stat_map(os.path.join(OUTPUT ,"pval_stat_treatment_effect_correction_Tmax.nii.gz"),\
                               colorbar=True,draw_cross=False,cut_coords=(-1,-13,14))

plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/treatment_pvalCorrected_map.png")




nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_treatment_effect_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
filemane = out_im
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_treatment_effect_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_treatment_effect_correction_Tmax.nii.gz"))


##################################################################################
