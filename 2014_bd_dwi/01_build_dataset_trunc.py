# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:01 2014

@author: christophe

concatenate FA images not skeletonised. Here the mask is truncated.
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

BASE_PATH = "/volatile/share/2014_bd_dwi"

INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_CSV = os.path.join(BASE_PATH, "population.csv")

INPUT_MASK = os.path.join(BASE_PATH,
                          "masks",
                          "mask_trunc.nii.gz")

OUTPUT_CSI = os.path.join(BASE_PATH, "bd_dwi_csi_trunc")
if not os.path.exists(OUTPUT_CSI):
    os.makedirs(OUTPUT_CSI)
OUTPUT_X = "X.npy"
OUTPUT_X_DESC = OUTPUT_X.replace("npy", "txt")
OUTPUT_Y = "Y.npy"
#############################################################################
## Build dataset

print "Reading data"
images4d = nib.load(INPUT_IMAGE)
image_arr = images4d.get_data()

population = pd.read_csv(INPUT_CSV,
                         index_col=0)
n, _ = population.shape

print "Reading mask"
babel_mask = nib.load(INPUT_MASK)
mask = babel_mask.get_data()
bin_mask = mask != 0
n_voxel_in_mask = np.count_nonzero(mask)
print "Number of voxels in mask:", n_voxel_in_mask
assert(n_voxel_in_mask == 448334)

# Create Y (same for all the cases)
Ytot = np.asarray(population.BD_HC, dtype='float64').reshape(n, 1)

# Apply mask to images & center
print "Application of mask on all the images & centering"
masked_images = np.zeros((n, n_voxel_in_mask))
for i, ID in enumerate(population.index):
    cur = population.iloc[i]
    slice_index = cur['SLICE']
    image = image_arr[:, :, :, slice_index]
    masked_images[i, :] = image[bin_mask]

masked_images -= masked_images.mean(axis=0)
masked_images /= masked_images.std(axis=0)

# Create categorial covariates
cat_covar_df = population[["SEX"]]
cat_covar = cat_covar_df.as_matrix()


# Create & center covariates
cont_covar = population[["AGEATMRI"]].as_matrix()
cont_covar -= cont_covar.mean(axis=0)
cont_covar /= cont_covar.std(axis=0)

covar = np.hstack((cont_covar, cat_covar))

# Case CS + Intercept
print "Saving CSI data"
Xtot = np.hstack([np.ones((covar.shape[0], 1)), covar, masked_images])
n, p = Xtot.shape
assert Xtot.shape == (n, n_voxel_in_mask + 3)
# X
np.save(os.path.join(OUTPUT_CSI, OUTPUT_X), Xtot)
# X_DESC
fh = open(os.path.join(OUTPUT_CSI, OUTPUT_X_DESC), "w")
fh.write('shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()
# Y
np.save(os.path.join(OUTPUT_CSI, OUTPUT_Y), Ytot)

#############################################################################