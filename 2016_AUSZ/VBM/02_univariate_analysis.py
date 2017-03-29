#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:06:35 2017

@author: ad247405
"""
import os
import numpy as np
import pandas as pd
import nibabel
from mulm import MUOLS
import nilearn
from nilearn import plotting

BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_CSV= os.path.join(BASE_PATH,"results","VBM","population.csv")
MASK_PATH = os.path.join(BASE_PATH,"results","VBM","data","mask.nii")

mask_ima = nibabel.load(MASK_PATH)
mask_arr = mask_ima.get_data() != 0

#asd vs controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","asd_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################



#scz-asd vs asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","scz_asd_vs_asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################


#scz_asd_vs_controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","scz_asd_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################



#scz_vs_asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","scz_vs_asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################




#scz_vs_controls
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","scz_vs_controls")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################


#scz_vs_scz-asd
##############################################################################
INPUT = os.path.join(BASE_PATH,"results","VBM","scz_vs_scz-asd")
OUTPUT = os.path.join(INPUT,"univariate_analysis")
X = np.load(os.path.join(INPUT , "X.npy"))
y = np.load(os.path.join(INPUT , "y.npy"))

Z = X[:, :3]
Y = X[: , 3:]
assert np.sum(mask_arr) == Y.shape[1]


DesignMat = np.zeros((Z.shape[0], Z.shape[1]+1)) # y, intercept, age, sex
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # sex


muols = MUOLS(Y=Y,X=DesignMat)
muols.fit() 
tvals, pvals, dfs = muols.t_test(contrasts=[1, 0, 0, 0], pval=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT, "pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT, "log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

#################################################################################