# -*- coding: utf-8 -*-
"""
@author: edouard.duchesnay@cea.fr

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT:
- mask.nii.gz
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import brainomics.image_atlas
import shutil

#import proj_classif_config
GENDER_MAP = {'Female': 0, 'Male': 1}

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs"


INPUT_CSV = os.path.join(BASE_PATH,          "ADRS", "population.csv")
OUTPUT = os.path.join(BASE_PATH,             "ADRS_csi")

if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)

# Read input subjects
# Read pop csv
pop = pd.read_csv(INPUT_CSV)
pop['Gender_num'] = pop["Gender"].map(GENDER_MAP)

#############################################################################
# Read images
n = len(pop)
assert n == 1082
Z = np.zeros((n, 3)) # Intercept + Age + Gender
Z[:, 0] = 1 # Intercept
y = np.zeros((n, 1)) # DX
images = list()
for i in xrange(n):
    cur = pop.iloc[i]
    print cur
    babel_image = nibabel.load(cur["mri_path"])
    images.append(babel_image.get_data().ravel())
    Z[i, 1:] = np.asarray(cur[["Age", "Gender_num"]]).ravel()
    y[i, 0] = cur["adrs"]

shape = babel_image.get_data().shape
np.save(os.path.join(OUTPUT, "y.npy"), y)

#############################################################################
# resample one anat
fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s" % \
("/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz", 
cur["mri_path"], 
os.path.join(OUTPUT, "MNI152_T1_1mm_brain.nii.gz"))

os.system(fsl_cmd)

#############################################################################
# Compute implicit mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
Xtot = np.vstack(images)
np.save(os.path.join(OUTPUT, "Xtot.npy"), Xtot)
del images

shape = babel_image.get_data().shape
#os.exit(0)
#Xtot = np.load(os.path.join(OUTPUT, "Xtot.npy"))

mask_implicit_arr = (np.min(Xtot, axis=0) > 0.01) & (np.std(Xtot, axis=0) > 1e-6)
mask_implicit_arr = mask_implicit_arr.reshape(shape)
assert mask_implicit_arr.sum() == 730646
out_im = nibabel.Nifti1Image(mask_implicit_arr.astype(int),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_implicit.nii.gz"))


#############################################################################
# Compute atlas mask
mask_atlas_ima = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=cur["mri_path"],
    output=os.path.join(OUTPUT, "mask_atlas.nii.gz"))

mask_atlas_arr = mask_atlas_ima.get_data()
assert np.sum(mask_atlas_arr != 0) == 638715
mask_atlas_arr[np.logical_not(mask_implicit_arr)] = 0  # apply implicit mask
# smooth
mask_atlas_arr = brainomics.image_atlas.smooth_labels(mask_atlas_arr, size=(3, 3, 3))
assert np.sum(mask_atlas_arr != 0) == 625897
out_im = nibabel.Nifti1Image(mask_atlas_arr,
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_atlas.nii.gz"))
im = nibabel.load(os.path.join(OUTPUT, "mask_atlas.nii.gz"))
assert np.all(mask_atlas_arr.astype(int) == im.get_data())


#############################################################################
# Compute mask with atlas but binarized (not group tv)
mask_atlas_binarized_arr = mask_atlas_arr != 0
assert mask_atlas_binarized_arr.sum() == 625897
out_im = nibabel.Nifti1Image(mask_atlas_binarized_arr.astype("int16"),
                             affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "mask_atlas_binarized.nii.gz"))
babel_mask = nibabel.load(os.path.join(OUTPUT, "mask_atlas_binarized.nii.gz"))
assert np.all(mask_atlas_binarized_arr == (babel_mask.get_data() != 0))

#############################################################################
# X
X = Xtot[:, mask_atlas_binarized_arr.ravel()]
#############################################################################
# BASIC MULM
from mulm import MUOLSStatsCoefficients
muols = MUOLSStatsCoefficients()
X_design = np.hstack([y, Z])
muols.fit(X_design, X)
#tvals, pvals = muols.stats(X_design, X)
tvals, pvals, dfs = muols.stats_t_coefficients(X=X_design, Y=X, contrast=[1, 0, 0, 0], pval=True)

# test the other side
tvals2, pvals2, dfs2 = muols.stats_t_coefficients(X=X_design, Y=X, contrast=[-1, 0, 0, 0], pval=True)
assert np.all(tvals == -tvals2)
assert np.all((pvals>.5) == (pvals2<.5))
assert np.allclose(1 - pvals[pvals>.5], pvals2[pvals>.5])
assert np.sum(tvals2>3) == 180
# End
pvals_2side = pvals.copy()
pvals_2side[pvals>.5] = 1 - pvals[pvals>.5]
assert np.allclose(pvals_2side[pvals>.5], pvals2[pvals>.5])
pvals_2side *= 2.

p_log10 = - np.log10(pvals_2side)
arr = np.zeros(shape)
arr[mask_atlas_binarized_arr] = p_log10
out_im = nibabel.Nifti1Image(arr, affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "pval_-log10_adrs.nii.gz"))

arr = np.zeros(shape)
arr[mask_atlas_binarized_arr] = tvals
out_im = nibabel.Nifti1Image(arr, affine=babel_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "tstat_adrs.nii.gz"))

import matplotlib.pyplot as plt
plt.plot(y, X[:, np.argmax(tvals)], "ob")
plt.savefig(os.path.join(OUTPUT, "gm-where-maxtval_x_adrs.svg"))
print "Max corr", np.corrcoef(y.ravel(), X[:, np.argmax(tvals)])
print "Min corr", np.corrcoef(y.ravel(), X[:, np.argmin(tvals)])

# ROI
# 70% Frontal Medial Cortex 25
#np.max(arr) 6.159

X = np.hstack([Z, X])
n, p = X.shape
X -= X.mean(axis=0)
X /= X.std(axis=0)
X[:, 0] = 1.
np.save(os.path.join(OUTPUT, "X.npy"), X)
fh = open(os.path.join(OUTPUT, "X.npy").replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask_atlas_binarized_arr.sum()))
fh.close()



