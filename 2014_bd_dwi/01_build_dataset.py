# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:01 2014

@author: christophe

concatenate FA images not skeletonised.
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

BASE_PATH = "/volatile/share/2014_bd_dwi"

INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_CSV = os.path.join(BASE_PATH, "population.csv")
OUTPUT_CS = os.path.join(BASE_PATH, "bd_dwi_cs")
if not os.path.exists(OUTPUT_CS):
    os.makedirs(OUTPUT_CS)
OUTPUT_CSI = os.path.join(BASE_PATH, "bd_dwi_csi")
if not os.path.exists(OUTPUT_CSI):
    os.makedirs(OUTPUT_CSI)
OUTPUT_X = "X.npy"
OUTPUT_X_DESC = OUTPUT_X.replace("npy", "txt")
OUTPUT_Y = "Y.npy"
OUTPUT_MASK_FILENAME = "mask.nii.gz"  # mask use to filtrate our images

FA_THRESHOLD = 0.05
# Image used to improve our mask
ATLAS_FILENAME = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz"
ATLAS_LABELS_RM = [0, 13, 2, 8]  # cortex, trunc

#############################################################################
## Build dataset

print "Reading data"
images4d = nib.load(INPUT_IMAGE)
image_arr = images4d.get_data()

population = pd.read_csv(INPUT_CSV,
                         index_col=0)
n, _ = population.shape

print "Computing mask"
## Threshold FA mean map
# to do if mask has not been registered yet
shape = image_arr.shape
fa_mean = np.mean(image_arr, axis=3)
mask = fa_mean > FA_THRESHOLD

# Remove cortex, trunc (thanks to atlas)
atlas = nib.load(ATLAS_FILENAME)
assert np.all(images4d.get_affine() == atlas.get_affine())
for label_rm in ATLAS_LABELS_RM:
    mask[atlas.get_data() == label_rm] = False
n_voxel_in_mask = np.count_nonzero(mask)
print "Number of voxels in mask:", n_voxel_in_mask
assert(n_voxel_in_mask == 507383)

# Store mask
out_im = nib.Nifti1Image(mask.astype(np.uint8), affine=images4d.get_affine())
out_im.to_filename(os.path.join(OUTPUT_CSI, OUTPUT_MASK_FILENAME))
out_im.to_filename(os.path.join(OUTPUT_CS, OUTPUT_MASK_FILENAME))

# Create Y (same for all the cases)
Ytot = np.asarray(population.BD_HC, dtype='float64').reshape(n, 1)

# Apply mask to images & center
print "Application of mask on all the images & centering"
masked_images = np.zeros((n, n_voxel_in_mask))
for i, ID in enumerate(population.index):
    cur = population.iloc[i]
    slice_index = cur['SLICE']
    image = image_arr[:, :, :, slice_index]
    masked_images[i, :] = image[mask]

masked_images -= masked_images.mean(axis=0)
masked_images /= masked_images.std(axis=0)

# Create & center covariates
covar = population[["AGEATMRI", "SEX"]].as_matrix()
covar -= covar.mean(axis=0)
covar /= covar.std(axis=0)

# Case CS
print "Saving CS data"
Xtot = np.hstack([covar, masked_images])
n, p = Xtot.shape
assert Xtot.shape == (n, n_voxel_in_mask + 2)
# X
np.save(os.path.join(OUTPUT_CS, OUTPUT_X), Xtot)
# X_DESC
fh = open(os.path.join(OUTPUT_CS, OUTPUT_X_DESC), "w")
fh.write('shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()
# Y
np.save(os.path.join(OUTPUT_CS, OUTPUT_Y), Ytot)

# Case CS + Intercept
print "Saving CSI data"
Xtot = np.hstack([np.ones((covar.shape[0], 1)), covar, masked_images])
n, p = Xtot.shape
assert Xtot.shape == (n, n_voxel_in_mask+3)
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
