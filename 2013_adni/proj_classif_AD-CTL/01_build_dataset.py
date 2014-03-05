# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:02:32 2014

@author: md238665

Read the non-smoothed images for all the subjects, mask them and dump them.
Similarly read group and dump it.

Data are then centered.

"""

# TODO: Images?

import os
import glob

import numpy as np
import pandas as pd

import sklearn.preprocessing as sklp

import nibabel

BASE_PATH = "/neurospin/brainomics/2013_adni"
PROJ_PATH = os.path.join(BASE_PATH,
                         "proj_classif")

INPUT_CLINIC_FILE = os.path.join(PROJ_PATH,
                                 "population.csv")
#INPUT_GROUPS = ['control', 'AD']

INPUT_TEMPLATE_PATH = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC")
INPUT_IMAGE_PATH = os.path.join(INPUT_TEMPLATE_PATH,
                                "registered_images")
INPUT_IMAGEFILE_FORMAT = os.path.join(INPUT_IMAGE_PATH,
                                      "mw{PTID}*_Nat_dartel_greyProba.nii")

INPUT_MASK_PATH = os.path.join(PROJ_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                          "mask.nii")

OUTPUT_PATH = PROJ_PATH
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_X_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.npy")
OUTPUT_MEAN_IMAGE = os.path.join(OUTPUT_PATH, "X_mean.nii")
OUTPUT_MEAN_MASKED_IMAGE = os.path.join(OUTPUT_PATH, "X_mean.masked.nii")
OUTPUT_X_TRAIN_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.train.npy")
OUTPUT_X_TEST_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.test.npy")
OUTPUT_X_TRAIN_CENTER_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.train.center.npy")
OUTPUT_X_TEST_CENTER_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.test.center.npy")
OUTPUT_X_MEAN_FILE = os.path.join(OUTPUT_PATH, "X_CTL_AD.mean.npy")
OUTPUT_Y_FILE = os.path.join(OUTPUT_PATH, "y_CTL_AD.npy")
OUTPUT_Y_TRAIN_FILE = os.path.join(OUTPUT_PATH, "y_CTL_AD.train.npy")
OUTPUT_Y_TEST_FILE = os.path.join(OUTPUT_PATH, "y_CTL_AD.test.npy")

# Read clinic data
pop = pd.read_csv(INPUT_CLINIC_FILE,
                  index_col=0)
n = len(pop)
print "Found", n, "subjects"

# Open mask
babel_mask = nibabel.load(INPUT_MASK)
mask = babel_mask.get_data() != 0
p = np.count_nonzero(mask)
print "Mask: {n} voxels".format(n=p)

# Load images & compute an average image (without mask)
print "Loading images"
X = np.zeros((n, p))
mean_image_data = np.zeros(mask.shape)
for i, PTID in enumerate(pop.index):
    #bv_group = m18_clinic_qc['Group.BV'].loc[PTID]
    #adni_group = m18_clinic_qc['Group.ADNI'].loc[PTID]
    print "Subject #{i} {PTID}, group {group}, {sample}".format(i=i,
                                                                PTID=PTID,
                                                                group=pop['Group.num'].loc[PTID],
                                                                sample=pop['Sample'].loc[PTID])
    imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=PTID)
    #print imagefile_pattern
    imagefile_name = glob.glob(imagefile_pattern)[0]
    babel_image = nibabel.load(imagefile_name)
    image_data = babel_image.get_data()
    # Apply mask (returns a flat image)
    X[i, :] = image_data[mask]
    # Store in mean image
    mean_image_data += image_data

# Store X and y
print "Storing data"
np.save(OUTPUT_X_FILE, X)
y = np.array(pop['Group.num'], dtype='float64')
np.save(OUTPUT_Y_FILE, y)

# Create mean image
mean_image_data /= n
mean_im = nibabel.Nifti1Image(mean_image_data, affine=babel_mask.get_affine())
mean_im.to_filename(OUTPUT_MEAN_IMAGE)

# Create average image (in mask)
X_mean_masked = np.zeros(mask.shape)
X_mean_masked[mask] = X.mean(axis=0)
X_mean_masked_im = nibabel.Nifti1Image(X_mean_masked, affine=babel_mask.get_affine())
X_mean_masked_im.to_filename(OUTPUT_MEAN_MASKED_IMAGE)

# Split in train-test according to Cuingnet et al. 2010
print "Splitting in train-test & storing"
train_subjects = pop['Sample'] == 'training'
test_subjects = ~train_subjects
X_train = X[train_subjects.nonzero()]
np.save(OUTPUT_X_TRAIN_FILE, X_train)
X_test  = X[test_subjects.nonzero()]
np.save(OUTPUT_X_TEST_FILE, X_test)

y_train = y[train_subjects.nonzero()]
np.save(OUTPUT_Y_TRAIN_FILE, y_train)
y_test  = y[test_subjects.nonzero()]
np.save(OUTPUT_Y_TEST_FILE, y_test)

# Centering X_train & X_test accordingly
print "Centering images & storing"
x_scaler = sklp.StandardScaler(with_std=False)
X_train_center = x_scaler.fit_transform(X_train)
np.save(OUTPUT_X_TRAIN_CENTER_FILE, X_train_center)
np.save(OUTPUT_X_MEAN_FILE, x_scaler.mean_)

X_test_center = x_scaler.transform(X_test)
np.save(OUTPUT_X_TEST_CENTER_FILE, X_test_center)
