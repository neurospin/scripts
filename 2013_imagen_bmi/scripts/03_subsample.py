# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:47:44 2013

@author: md238665

This script subsamples the Gaser-segmented images and the mask.

The segmentation procedure is borrowed from Vincent Frouin subsampling scripts.
It uses a recent version of nilearn.
The git repository can be cloned at https://github.com/nilearn/nilearn.git

The processing are made for smoothed and non-smoothed images.

Warning: there are 1534 images but only 1265 subjects for BMI.
We subsample all the images.

Warning: the convention for new file names is very weak.

Warning: we output files in the data/ directory.

"""

import os
import numpy, pandas
import nibabel
from nilearn.image.resampling import resample_img

import bmi_utils

# Input
BASE_DIR = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_DIR = os.path.join(BASE_DIR, 'data')
CLINIC_DIR = os.path.join(DATA_DIR, 'clinic')
IMAGES_DIR = os.path.join(DATA_DIR, 'VBM', 'gaser_vbm8')
IMG_FILENAME_TEMPLATE = 'mwp1{subject_id:012}*.nii'
SMOOTHED_IMG_FILENAME_TEMPLATE = 'smwp1{subject_id:012}*.nii'
FULL_CLINIC_FILE = os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv')

# Original mask and mask without cerebellum
MASK_FILE = os.path.join(DATA_DIR, 'mask', 'mask.nii')
MASK_WITHOUT_CEREBELUM_FILE = os.path.join(DATA_DIR, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii')

# Images will be shaped as this image
TARGET_FILE = '/neurospin/brainomics/neuroimaging_ressources/atlases/HarvardOxford/HarvardOxford-LR-cort-333mm.nii.gz'

# Output files
RESULTS_DIR = os.path.join(BASE_DIR, 'data')
COMMON_OUT_DIR = os.path.join(RESULTS_DIR, 'subsampled_images')
#OUT_IMG_FILENAME_TEMPLATE = 'rmwp1{subject_id:012}*.nii'
IMG_OUT_DIR = os.path.join(COMMON_OUT_DIR, 'non_smoothed')
#OUT_SMOOTHED_IMG_FILENAME_TEMPLATE = 'rsmwp1{subject_id:012}*.nii'
SMOOTHED_IMG_OUT_DIR = os.path.join(COMMON_OUT_DIR, 'smoothed')
RMASK_FILE = os.path.join(COMMON_OUT_DIR, 'rmask.nii')
RMASK_WITHOUT_CEREBELUM_FILE = os.path.join(COMMON_OUT_DIR, 'rmask_without_cerebellum_7.nii')
if not os.path.exists(COMMON_OUT_DIR):
    os.makedirs(COMMON_OUT_DIR)
if not os.path.exists(IMG_OUT_DIR):
    os.makedirs(IMG_OUT_DIR)
if not os.path.exists(SMOOTHED_IMG_OUT_DIR):
    os.makedirs(SMOOTHED_IMG_OUT_DIR)

# Subsampling function
def resample(input_image, target_affine, target_shape, interpolation='continuous'):
    input_image_data = input_image.get_data()
    nan_mask = numpy.isnan(input_image_data)
    if numpy.any(nan_mask):
        input_image_data[nan_mask] = 0.0
        input_image = nibabel.Nifti1Image(input_image_data, input_image.get_affine())
    outim = resample_img(input_image,
                         target_affine=target_affine, target_shape=target_shape,
                         interpolation=interpolation)
    return outim

# Open the clinic file to get all the subject's ID
df = pandas.io.parsers.read_csv(FULL_CLINIC_FILE)
all_subjects_indices = df.Subjects
n_images = df.shape[0]

# Open target image
target = nibabel.load(TARGET_FILE)
target_shape = target.shape
target_affine = target.get_affine()

# Subsample masks & save them
babel_mask  = nibabel.load(MASK_FILE)
babel_rmask = resample(babel_mask, target_affine, target_shape, interpolation='continuous')
nibabel.save(babel_rmask, RMASK_FILE)
rmask        = babel_rmask.get_data()
binary_rmask = rmask!=0
useful_voxels = numpy.ravel_multi_index(numpy.where(binary_rmask), rmask.shape)
n_useful_voxels = len(useful_voxels)
print "Subsampled {i} to {o}".format(i=MASK_FILE,
                                     o=RMASK_FILE)
print "Subsampled mask: {n} true voxels".format(n=n_useful_voxels)

babel_mask_without_cerebellum  = nibabel.load(MASK_WITHOUT_CEREBELUM_FILE)
babel_rmask_without_cerebellum = resample(babel_mask_without_cerebellum, target_affine, target_shape, interpolation='continuous')
nibabel.save(babel_rmask_without_cerebellum, RMASK_WITHOUT_CEREBELUM_FILE)
rmask_without_cerebellum        = babel_rmask_without_cerebellum.get_data()
binary_rmask_without_cerebellum = rmask_without_cerebellum!=0
useful_voxels_without_cerebellum = numpy.ravel_multi_index(numpy.where(binary_rmask_without_cerebellum), rmask_without_cerebellum.shape)
n_useful_voxels_without_cerebellum = len(useful_voxels_without_cerebellum)
print "Subsampled {i} to {o}".format(i=MASK_WITHOUT_CEREBELUM_FILE,
                                     o=RMASK_WITHOUT_CEREBELUM_FILE)
print "Subsampled mask without cerebellum: {n} true voxels".format(n=n_useful_voxels_without_cerebellum)

#
# Subsample non-smoothed images
#
all_images_filename = bmi_utils.find_images(all_subjects_indices, IMG_FILENAME_TEMPLATE, IMAGES_DIR)
print "Found {l} non-smoothed images".format(l=len(all_images_filename))

# Read images in the same order than subjects & subsample
for (index, full_in_filename) in enumerate(all_images_filename):
    filename = os.path.basename(full_in_filename)
    # Generate output name
    full_out_filename = os.path.join(IMG_OUT_DIR, 'r'+filename)
    print "Subsampling {i} to {o}".format(i=full_in_filename, o=full_out_filename)
    # Subsample and save
    input_image = nibabel.load(full_in_filename)
    out_image = resample(input_image, target_affine, target_shape)
    nibabel.save(out_image, full_out_filename)
print "Non-smoothed images subsampled"

#
# Subsample smoothed images
#
all_smoothed_images_filename = bmi_utils.find_images(all_subjects_indices, SMOOTHED_IMG_FILENAME_TEMPLATE, IMAGES_DIR)
print "Found {l} non-smoothed images".format(l=len(all_smoothed_images_filename))

# Read images in the same order than subjects & subsample
for (index, full_in_filename) in enumerate(all_smoothed_images_filename):
    filename = os.path.basename(full_in_filename)
    # Generate output name
    full_out_filename = os.path.join(SMOOTHED_IMG_OUT_DIR, 'r'+filename)
    print "Subsampling {i} to {o}".format(i=full_in_filename, o=full_out_filename)
    # Subsample and save
    input_image = nibabel.load(full_in_filename)
    out_image = resample(input_image, target_affine, target_shape)
    nibabel.save(out_image, full_out_filename)
print "Smoothed images subsampled"
