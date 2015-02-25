# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 14:00:06 2014

@author: cp243490



"""

import os
import numpy as np
import nibabel as nib

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs/"

DATA_PATH = os.path.join(BASE_PATH,             "ADRS_datasets")
OUTPUT_UNIVARIATE = os.path.join(BASE_PATH,     "ADRS_univariate")

if not os.path.exists(OUTPUT_UNIVARIATE):
    os.makedirs(OUTPUT_UNIVARIATE)
#############################################################################
## Read data


mask_ima = nib.load(os.path.join(DATA_PATH, "mask_atlas_binarized.nii.gz"))
mask_arr = mask_ima.get_data() != 0
y = np.load(os.path.join(DATA_PATH, "y.npy"))
X = np.load(os.path.join(DATA_PATH, "X.npy"), mmap_mode='r')
Z = X[:, :3]
X = X[:, 3:]

#assert np.sum(mask_arr) == Y.shape[1]

print "ok read"

#############################################################################
## BASIC MULM
DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
DesignMat[:, 1] = 1  # intercept
DesignMat[:, 2] = Z[:, 1]  # age
DesignMat[:, 3] = Z[:, 2]  # gender


contrasts = np.array([[1, 0, 0, 0]])

from mulm import MUOLS
muols = MUOLS(Y=X, X=DesignMat)
muols.fit(block=True)
print "ok fit"
tvals, pvals, dfs = muols.t_test(contrasts=contrasts,
                                 pval=True,
                                 two_tailed=True)
print "ok t_test"
del muols
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE, "pval_adrs.nii.gz"))

p_log10 = - np.log10(pvals)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = p_log10[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE, "pval_-log10_adrs.nii.gz"))

arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE, "tstat_adrs.nii.gz"))

#X = np.load(os.path.join(DATA_PATH, "X.npy"))
#import matplotlib.pyplot as plt
#plt.plot(y, X[:, np.argmax(tvals)], "ob")
#plt.savefig(os.path.join(OUTPUT_UNIVARIATE, "gm-where-maxtval_x_adrs.svg"))
#print "Max corr", np.corrcoef(y.ravel(), X[:, np.argmax(tvals)])
#print "Min corr", np.corrcoef(y.ravel(), X[:, np.argmin(tvals)])

print "ok univariate analysis"
###########################################
#### Univariate Analysis with permutations
##########################################
#
##Univariate analysis with empirical pvalues corrected by permutations
## Permutation procedure to correct the pvalue: maxT
#
nperms = 100
#################################
# Correctied pvalue for the brain
muols = MUOLS(Y=X, X=DesignMat)
muols.fit(block=True)
print "ok fit permutations"
tvals, pvals_perm, _ = muols.t_test_maxT(contrasts=contrasts,
                                    nperms=nperms,
                                    two_tailed=True)
print "ok maxT test permutations"
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = tvals[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
               "tstat_adrs.nii.gz"))

arr = np.zeros(mask_arr.shape)
arr[mask_arr] = pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
         "ptmax_adrs.nii.gz"))

log10_pvals_perm = -np.log10(pvals_perm)
arr = np.zeros(mask_arr.shape)
arr[mask_arr] = log10_pvals_perm[0]
out_im = nib.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE,
         "ptmax-log10_adrs.nii.gz"))
del muols
print "ok permutations"