# -*- coding: utf-8 -*-
"""
@author: edouard.Duchesnay@cea.fr

Compute variability over weights maps

INPUT:
- mask.nii
- glob pattern directory/*/1.1

OUTPUT: 

"""

import os
import numpy as np
import glob
import nibabel
import gzip

BASE_PATH = "/neurospin/brainomics/2013_adni"
#INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_baseline.csv")

#WHICH = '0.01_0.0_0.999_0.001'
#WHICH = '0.01_0.0_1.0_0.0'
WHICH = '0.01_0.01_0.99_0.0'
WHICH = '0.01_0.00999_0.98901_0.001'

INPUT_MASK = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "mask.nii")
INPUT_PATTERN = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc",
    'logistictvenet_5cv/results/*/%s/beta.npy.gz' % WHICH)

OUTPUT_DIR = os.path.join(BASE_PATH, "proj_classif_MCIc-MCInc", "weights")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

mask_image = nibabel.load(INPUT_MASK)
mask = mask_image.get_data() != 0

files = glob.glob(INPUT_PATTERN)

W = list()
for f in files:
    W.append(np.load(gzip.open(f))[3:].ravel())

W = np.vstack(W)

n2 = np.sqrt(np.sum(W ** 2, axis=1))[:, np.newaxis]
W = W / n2 
print "norm 2 =", np.sqrt(np.sum(W ** 2, axis=1))

sd = np.std(W, axis=0)
mean = np.mean(W, axis=0)
sd.max()


arr = np.zeros(mask.shape)

arr[mask] = sd
out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DIR, WHICH + "_sd.nii"))

arr[mask] = mean
out_im = nibabel.Nifti1Image(arr,affine=mask_image.get_affine())
out_im.to_filename(os.path.join(OUTPUT_DIR, WHICH + "_mean.nii"))
