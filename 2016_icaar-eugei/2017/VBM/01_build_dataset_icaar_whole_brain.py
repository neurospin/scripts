# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:25:41 2016

@author: ad247405

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT_ICAARZ:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import pandas as pd
import nibabel
import brainomics.image_atlas
import mulm
import nilearn
from nilearn import plotting
from mulm import MUOLS

#import proj_classif_config
GENDER_MAP = {'F': 0, 'H': 1}

BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei'
INPUT_CSV_ICAAR = os.path.join(BASE_PATH,"VBM","ICAAR","population.csv")
OUTPUT_UNIVARIATE_ICAAR = os.path.join(BASE_PATH,"VBM","ICAAR","results","univariate")

OUTPUT_DATA_ICAAR = os.path.join(BASE_PATH,"VBM","ICAAR","data")

# Read pop csv
pop = pd.read_csv(INPUT_CSV_ICAAR)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
#############################################################################
# Read images
n = len(pop)
assert n == 55
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i, index in enumerate(pop.index):
    cur = pop[pop.index== index]
    print(cur)
    imagefile_name = cur.path_VBM
    babel_image = nibabel.load(imagefile_name.as_matrix()[0])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["age", "sex.num"]]).ravel()
    y[i, 0] = cur["group_outcom.num"]

shape = babel_image.get_data().shape

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
mask = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask = mask.reshape(shape)
assert mask.sum() == 345041

#############################################################################
# Compute atlas mask
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name.as_matrix()[0],
    output=os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 617728
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 301440
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
im = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
assert np.all(mask_atlas == im.get_data())

#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_bool = mask_atlas != 0
mask_bool.sum() == 301440
out_im = nibabel.Nifti1Image(mask_bool.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
babel_mask = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
assert np.all(mask_bool == (babel_mask.get_data() != 0))


#############################################################################

# Save data X and y
X = Xtot[:, mask_bool.ravel()]
#Use mean imputation, we could have used median for age
#imput = sklearn.preprocessing.Imputer(strategy = 'median',axis=0)
#Z = imput.fit_transform(Z)
X = np.hstack([Z, X])
assert X.shape == (55, 301443)

#Remove nan lines
X= X[np.logical_not(np.isnan(y)).ravel(),:]
y=y[np.logical_not(np.isnan(y))]
assert X.shape == (41, 301443)


X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
n, p = X.shape
X = np.nan_to_num(X)
np.save(os.path.join(OUTPUT_DATA_ICAAR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DATA_ICAAR, "y.npy"), y)

###############################################################################
# precompute linearoperator
X = np.load(os.path.join(OUTPUT_DATA_ICAAR, "X.npy"))
y = np.load(os.path.join(OUTPUT_DATA_ICAAR, "y.npy"))

mask = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))

import parsimony.functions.nesterov.tv as nesterov_tv
from parsimony.utils.linalgs import LinearOperatorNesterov

Atv = nesterov_tv.linear_operator_from_mask(mask.get_data(), calc_lambda_max=True)
Atv.save(os.path.join(OUTPUT_DATA_ICAAR, "Atv.npz"))
Atv_ = LinearOperatorNesterov(filename=os.path.join(OUTPUT_DATA_ICAAR, "Atv.npz"))
assert Atv.get_singular_values(0) == Atv_.get_singular_values(0)



###############################################################################
#############################################################################
# MULM
mask_ima = nibabel.load(os.path.join(OUTPUT_DATA_ICAAR, "mask_whole_brain.nii"))
mask_arr = mask_ima.get_data() != 0
X = np.load(os.path.join(OUTPUT_DATA_ICAAR, "X.npy"))
y = np.load(os.path.join(OUTPUT_DATA_ICAAR, "y.npy"))
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
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"t_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"pval_stat_conversion_NoConversion.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"log10pval_stat_conversion_NoConversion.nii.gz"))


nperms = 1000
tvals_perm, pvals_perm, _ = muols.t_test_maxT(contrasts=np.array([[1, 0, 0, 0]]),nperms=nperms,two_tailed=True)
arr = np.zeros(mask_arr.shape); arr[mask_arr] = tvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = pvals_perm[0]
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))

arr = np.zeros(mask_arr.shape); arr[mask_arr] = -np.log10(pvals_perm[0])
out_im = nibabel.Nifti1Image(arr, affine=mask_ima.get_affine())
out_im.to_filename(os.path.join(OUTPUT_UNIVARIATE_ICAAR,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz"))



#Plot univariate analysis map
filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"t_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"log10pval_stat_conversion_NoConversion.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - uncorrected")

filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"t_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,title = "T-statistic map")

filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False)

filename = os.path.join(OUTPUT_UNIVARIATE_ICAAR,"log10pval_stat_conversion_NoConversion_correction_Tmax.nii.gz")
nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold=2,title = "log10(pvalue) - corrected for multiple comparisons")

##################################################################################
