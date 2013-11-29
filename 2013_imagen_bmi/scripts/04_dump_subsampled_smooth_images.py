# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:57:03 2013

@author: md238665

Create an HDF5 used for fast access.
Here we store:
 - the subject ID, the SNPs and the BMI data
 - the subsampled smoothed images (standard mask)
 - the subsampled smoothed images (mask without cerebellum)

The order is read from the file subject_id.csv.

TODO:
 - write all covariates?

Warning: existing file will be deleted

"""

import os, sys
import numpy, pandas
import tables

import nibabel

import bmi_utils

########################
# Input & output files #
########################
DATA_PATH = '/neurospin/brainomics/2013_imagen_bmi/data'
INPUT_SUBJECTS_ID_FILE = os.path.join(DATA_PATH, 'subjects_id.csv')

# R does not deal well wit int64 so we use int32
subjects_id = pandas.io.parsers.read_csv(INPUT_SUBJECTS_ID_FILE, squeeze=True).astype('int32')
subjects_id_index = pandas.Index(subjects_id, name='subject_id')

INPUT_SNP_FILE = os.path.join(DATA_PATH, 'SNPs.csv')
INPUT_BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

OUTPUT_FILE = os.path.join(DATA_PATH, 'subsampled_smoothed_images.hdf5')
if os.path.exists(OUTPUT_FILE):
    print "Warning: continuing will erase %s" % OUTPUT_FILE
    a = raw_input("Are you sure [Y/n]?")
    if a != 'Y':
        sys.exit()
h5file = tables.openFile(OUTPUT_FILE, mode="w")

#
# Read & store the subject IDs, the SNPs and the BMI
#
bmi_utils.store_array(h5file, subjects_id, "subject_id")
# SNPs will be used as doubles so we convert them
SNPs   = pandas.io.parsers.read_csv(INPUT_SNP_FILE, index_col=0).astype(numpy.float64)
bmi_utils.store_array(h5file, SNPs, 'SNPs')
clinic = pandas.io.parsers.read_csv(INPUT_BMI_FILE, index_col=0)
bmi_utils.store_array(h5file, clinic, 'BMI')
print "Subject ID, SNPs and BMI data dumped"

#
# Find images
#
IMG_PATH  = os.path.join(DATA_PATH, 'subsampled_images', 'smoothed')
IMG_FILENAME_TEMPLATE = 'rsmwp1{subject_id:012}*.nii'
sub_smoothed_files = bmi_utils.find_images(subjects_id, IMG_FILENAME_TEMPLATE, IMG_PATH)
print "Found", len(sub_smoothed_files), "images"

#
# Read & store smoothed data with standard mask
#
MASK_PATH = os.path.join(DATA_PATH, 'subsampled_images', 'rmask.nii')
babel_mask  = nibabel.load(MASK_PATH)

sub_smoothed_images = bmi_utils.read_images_with_mask(sub_smoothed_files, babel_mask)
bmi_utils.store_images_and_mask(h5file, sub_smoothed_images, babel_mask, group_name="subsampled_smoothed_images")
print "Subsampled smoothed images dumped"
del sub_smoothed_images

#
# Read & store smoothed data without cerebellum mask
#
MASK_PATH = os.path.join(DATA_PATH, 'subsampled_images', 'rmask_without_cerebellum_7.nii')
babel_mask_without_cerebellum  = nibabel.load(MASK_PATH)

sub_smoothed_images_without_cerebellum = bmi_utils.read_images_with_mask(sub_smoothed_files, babel_mask_without_cerebellum)
bmi_utils.store_images_and_mask(h5file, sub_smoothed_images_without_cerebellum, babel_mask_without_cerebellum, group_name="subsampled_smoothed_images_without_cerebellum")
print "Subsampled smoothed images without cerebellum dumped"
del sub_smoothed_images_without_cerebellum

h5file.close()