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


OUTPUT_MASK = os.path.join(BASE_PATH, EXPERIMENT, "mask.nii")
OUTPUT_X_TRAIN = os.path.join(BASE_PATH,  EXPERIMENT, "GMtrain.npy")
OUTPUT_y_TRAIN = os.path.join(BASE_PATH, EXPERIMENT, "ytrain.npy")
OUTPUT_X_TEST = os.path.join(BASE_PATH,  EXPERIMENT, "GMtest.npy")
OUTPUT_SUBJ_TEST = os.path.join(BASE_PATH,  EXPERIMENT, "test_subjects.csv")
if not os.path.exists(os.path.join(BASE_PATH, EXPERIMENT)):
        os.makedirs(os.path.join(BASE_PATH, EXPERIMENT))

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

print "Train", len(images_train)
print "Test", len(images_test)

##############################################################################
## Read Test images
#images_test = list()
#sid_test = list()
#for test_filename in glob.glob(INPUT_IMAGEFILE_FORMAT_TEST):
#    print test_filename
#    #test_filename = '/neurospin/tmp/mlc2014/processed/binary/Test_Sbj1/mwrc1Test_Sbj1.nii.gz'
#    babel_image = nibabel.load(test_filename)
#    images_test.append(babel_image.get_data().ravel())
#    SID = re.findall("Sbj[0-9]+", test_filename)[0]
#    sid_test.append(SID)
#
#print "Test",  len(images_test)

#    imagefile_pattern_test = INPUT_IMAGEFILE_FORMAT_TEST.format(SID=SID)
#    imagefile_name_test = glob.glob(imagefile_pattern_test)

#############################################################################
# Compute mask
# Implicit Masking involves assuming that a lower than a givent threshold 
# at some voxel, in any of the images, indicates an unknown and is
# excluded from the analysis.
print ""
Xtot = np.vstack(images_train + images_test)
mask = np.min(Xtot, axis=0) > 0.01
print "nvox =", mask.sum()
# nvox = 379777

shape = babel_image.get_data().shape
# Save mask
out_im = nibabel.Nifti1Image(mask.astype(int).reshape(shape),
                             affine=babel_image.get_affine())
out_im.to_filename(OUTPUT_MASK)
babel_mask = nibabel.load(OUTPUT_MASK)
assert np.all(mask == (babel_mask.get_data() != 0).ravel())

#############################################################################
# X
Ztrain = np.zeros((len(images_train), 1)) # Intercept + Age + Gender
Ztrain[:, 0] = 1 # Intercept
Xtrain = np.vstack(images_train)[:, mask]
Xtrain = np.hstack([Ztrain, Xtrain])
print "# Train X.shape =", Xtrain.shape
n, p = Xtrain.shape
np.save(OUTPUT_X_TRAIN, Xtrain)
fh = open(OUTPUT_X_TRAIN.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept + %i voxels' % \
    (n, p, mask.sum()))
fh.close()
np.save(OUTPUT_y_TRAIN, y_train)

#X.shape = (150, 365620)
Ztest = np.zeros((len(images_test), 1)) # Intercept + Age + Gender
Ztest[:, 0] = 1 # Intercept
Xtest = np.vstack(images_test)[:, mask]
Xtest = np.hstack([Ztest, Xtest])
print "# Test X.shape =", Xtest.shape
n, p = Xtest.shape
np.save(OUTPUT_X_TEST, Xtest)
fh = open(OUTPUT_X_TEST.replace("npy", "txt"), "w")
fh.write('shape = (%i, %i): Intercept +  %i voxels' % \
    (n, p, mask.sum()))
fh.close()
