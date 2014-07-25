# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:01 2014

@author: christophe
"""
import os
import nibabel as nib
import numpy as np
import pandas as pd

BASE_PATH =  "/volatile/share/2014_bd_dwi"

INPUT_IMAGE = os.path.join(BASE_PATH, "all_FA/nii/stats/all_FA.nii.gz")
INPUT_CSV = os.path.join(BASE_PATH, "population.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "bd_dwi_cs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_X = os.path.join(OUTPUT_DIR, "X.npy")
OUTPUT_X_DESC = OUTPUT_X.replace("npy", "txt")
OUTPUT_Y = os.path.join(OUTPUT_DIR, "Y.npy")
OUTPUT_MASK_FILENAME =  os.path.join(OUTPUT_DIR, "mask.nii.gz") #mask use to filtrate our images

FA_THRESHOLD = 0.05
ATLAS_FILENAME = "/usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz" # Image use to improve our mask
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

#registration of mask
out_im = nib.Nifti1Image(mask.astype(int), affine=images4d.get_affine())
out_im.to_filename(OUTPUT_MASK_FILENAME)
# Check that the stored image is the same than mask
babel_mask = nib.load(OUTPUT_MASK_FILENAME)
assert np.all(mask == (babel_mask.get_data() != 0))
print "Number of voxels in mask:", mask.sum()

n_voxel_in_mask = np.count_nonzero(mask)
assert(n_voxel_in_mask == 507383)

print "Application of mask on all the images"
Ytot = np.asarray(population.BD_HC, dtype='float64').reshape(n, 1)
Ztot = population[["AGEATMRI", "SEX"]].as_matrix()

Xtot = np.zeros((n, n_voxel_in_mask))
for i, ID in enumerate(population.index):
    cur = population.iloc[i]
    slice_index = cur.SLICE
    image = image_arr[:, :, :, slice_index]
    Xtot[i, :] = image[mask]

Xtot = np.hstack([Ztot, Xtot])
assert Xtot.shape == (n, n_voxel_in_mask+2)

print "Centering and scaling data"
Xtot -= Xtot.mean(axis = 0)
Xtot /= Xtot.std(axis = 0)

print "Saving data"
n, p = Xtot.shape
np.save(OUTPUT_X, Xtot)
fh = open(OUTPUT_X_DESC, "w")
fh.write('shape = (%i, %i): Age + Gender + %i voxels' % \
    (n, p, mask.sum()))
fh.close()

np.save(OUTPUT_Y, Ytot)

#############################################################################
