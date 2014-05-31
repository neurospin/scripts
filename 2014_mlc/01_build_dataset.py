# -*- coding: utf-8 -*-
"""
@author: edouard.Duchesnay@cea.fr

Compute mask, concatenate masked non-smoothed images for all the subjects.
Build X, y, and mask

INPUT:
- subject_list.txt:
- population.csv

OUTPUT:
- mask.nii
- y.npy
- X.npy = intercept + Age + Gender + Voxel
"""

import os
import numpy as np
import glob
import pandas as pd
import nibabel
import re
#import proj_classif_config

BASE_PATH =  "/neurospin/brainomics/2014_mlc"
IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
EXPERIMENT = "GM"
#INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_baseline.csv")

INPUT_IMAGEFILE_FORMAT_TRAIN = os.path.join(IMAGES_PATH,
                                   "Train_{SID}",
                                   "mwrc1*.nii.gz")
INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")

INPUT_IMAGEFILE_FORMAT_TEST = os.path.join(IMAGES_PATH,
                                   "Test_{SID}",
                                   "mwrc1*.nii.gz")
INPUT_SUBJECT_TEST = os.path.join(IMAGES_PATH, "BinaryTest_sbj_list.csv")
INPUT_ROI_TEST = os.path.join(IMAGES_PATH, "BinaryTest_data.csv")


OUTPUT_MASK = os.path.join(BASE_PATH, "GM", "mask.nii.gz")
OUTPUT_X_TRAIN = os.path.join(BASE_PATH,  "GM", "GMtrain.npy")
OUTPUT_y_TRAIN = os.path.join(BASE_PATH, "GM", "ytrain.npy")
OUTPUT_X_TEST = os.path.join(BASE_PATH,  "GM", "GMtest.npy")
OUTPUT_SUBJ_TEST = os.path.join(BASE_PATH,  "GM", "test_subjects.csv")
if not os.path.exists(os.path.join(BASE_PATH, "GM")):
        os.makedirs(os.path.join(BASE_PATH, "GM"))

OUTPUT_MASK_ATLAS = os.path.join(BASE_PATH, "gm_gtvenet", "mask_atlas.nii.gz")
OUTPUT_X_TRAIN_ATLAS = os.path.join(BASE_PATH, "gm_gtvenet", "Xtrain_atlas.npy")
OUTPUT_X_TEST_ATLAS = os.path.join(BASE_PATH, "gm_gtvenet", "Xtest_atlas.npy")

if not os.path.exists(os.path.join(BASE_PATH, "gm_gtvenet")):
        os.makedirs(os.path.join(BASE_PATH, "gm_gtvenet"))

# Read pop csv
subject_train = pd.read_csv(INPUT_SUBJECT_TRAIN)
roi_train = pd.read_csv(INPUT_ROI_TRAIN, header=None)
subject_test = pd.read_csv(INPUT_SUBJECT_TEST)
roi_test = pd.read_csv(INPUT_ROI_TEST, header=None)

#############################################################################
# Read Train images
n = len(subject_train)
y_train = np.zeros((n, 1)) # DX

images_train = list()
for i, SID in enumerate(subject_train.SID):
    cur = subject_train.iloc[i]
    print cur
    imagefile_pattern_train = INPUT_IMAGEFILE_FORMAT_TRAIN.format(SID=cur["SID"])
    imagefile_name_train = glob.glob(imagefile_pattern_train)
    if len(imagefile_name_train) != 1:
        raise ValueError("Found %i files" % len(imagefile_name_train))
    babel_image = nibabel.load(imagefile_name_train[0])
    images_train.append(babel_image.get_data().ravel())
    y_train[i, 0] = cur["Label"]

#############################################################################
# Read Test images
images_test = list()
for i, SID in enumerate(subject_test.SID):
    cur = subject_test.iloc[i]
    print cur
    imagefile_pattern_test = INPUT_IMAGEFILE_FORMAT_TRAIN.format(SID=cur["SID"])
    imagefile_name_test = glob.glob(imagefile_pattern_test)
    if len(imagefile_name_test) != 1:
        raise ValueError("Found %i files" % len(imagefile_name_test))
    babel_image = nibabel.load(imagefile_name_test[0])
    images_test.append(babel_image.get_data().ravel())

assert len(images_train) == 150
assert len(images_test) == 100


#############################################################################
# Compute implicit mask
# Implicit Masking involves assuming that a lower than a givent threshold
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
shape = babel_image.get_data().shape

Xtot = np.vstack(images_train + images_test)
mask = np.min(Xtot, axis=0) > 0.01
mask = mask.reshape(shape)
assert mask.sum() == 379777

# Save mask
out_im = nibabel.Nifti1Image(mask.astype(int),
                             affine=babel_image.get_affine())
out_im.to_filename(OUTPUT_MASK)
babel_mask = nibabel.load(OUTPUT_MASK)
assert np.all(mask == (babel_mask.get_data() != 0))

#############################################################################
# X
Ztrain = np.zeros((len(images_train), 1)) # Intercept
Ztrain[:, 0] = 1 # Intercept
Xtrain = np.vstack(images_train)[:, mask.ravel()]
Xtrain = np.hstack([Ztrain, Xtrain])
assert Xtrain.shape == (150, 379778)
n, p = Xtrain.shape
np.save(OUTPUT_X_TRAIN, Xtrain)
fh = open(OUTPUT_X_TRAIN.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + %i voxels' % \
    (n, p, mask.sum()))
fh.close()
np.save(OUTPUT_y_TRAIN, y_train)

#X.shape = (150, 365620)
Ztest = np.zeros((len(images_test), 1)) # Intercept
Ztest[:, 0] = 1 # Intercept
Xtest = np.vstack(images_test)[:, mask.ravel()]
Xtest = np.hstack([Ztest, Xtest])
assert Xtest.shape == (100, 379778)
n, p = Xtest.shape
np.save(OUTPUT_X_TEST, Xtest)
fh = open(OUTPUT_X_TEST.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept +  %i voxels' % \
    (n, p, mask.sum()))
fh.close()


#############################################################################
# Compute atlas mask
import brainomics.image_atlas
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(
    ref=imagefile_name_test[0],
    output=OUTPUT_MASK_ATLAS)

mask_atlas = babel_mask_atlas.get_data()
assert np.sum(mask_atlas != 0) == 638715
mask_atlas[np.logical_not(mask)] = 0  # apply implicit mask
# smooth
mask_atlas = brainomics.image_atlas.smooth_labels(mask_atlas, size=(3, 3, 3))
assert np.sum(mask_atlas != 0) == 340991
out_im = nibabel.Nifti1Image(mask_atlas,
                             affine=babel_image.get_affine())
out_im.to_filename(OUTPUT_MASK_ATLAS)
im = nibabel.load(OUTPUT_MASK_ATLAS)
assert np.all(mask_atlas == im.get_data())
mask_atlas = mask_atlas != 0


#############################################################################
# X
Ztrain = np.zeros((len(images_train), 1)) # Intercept
Ztrain[:, 0] = 1 # Intercept
Xtrain = np.vstack(images_train)[:, mask_atlas.ravel()]
Xtrain = np.hstack([Ztrain, Xtrain])
assert Xtrain.shape == (150, 340992)
n, p = Xtrain.shape
np.save(OUTPUT_X_TRAIN_ATLAS, Xtrain)
fh = open(OUTPUT_X_TRAIN_ATLAS.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + %i voxels' % \
    (n, p, mask_atlas.sum()))
fh.close()

#X.shape = (150, 365620)
Ztest = np.zeros((len(images_test), 1)) # Intercept
Ztest[:, 0] = 1 # Intercept
Xtest = np.vstack(images_test)[:, mask_atlas.ravel()]
Xtest = np.hstack([Ztest, Xtest])
assert Xtest.shape == (100, 340992)
n, p = Xtest.shape
np.save(OUTPUT_X_TEST_ATLAS, Xtest)
fh = open(OUTPUT_X_TEST_ATLAS.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept +  %i voxels' % \
    (n, p, mask_atlas.sum()))
fh.close()


#############################################################################
#  Check nans
Xtrain = np.load(OUTPUT_X_TRAIN)
Xtest = np.load(OUTPUT_X_TEST)
assert Xtrain.shape == (150, 379778)
assert Xtest.shape == (100, 379778)
assert np.sum(np.isnan(Xtrain)) == 0
assert np.sum(np.isnan(Xtest)) == 0


Xtrain = np.load(OUTPUT_X_TRAIN_ATLAS)
Xtest = np.load(OUTPUT_X_TEST_ATLAS)
assert Xtrain.shape == (150, 340992)
assert Xtest.shape == (100, 340992)
assert np.sum(np.isnan(Xtrain)) == 0
assert np.sum(np.isnan(Xtest)) == 0
