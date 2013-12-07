# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:57:03 2013

@author: md238665

Create an HDF5 file used for fast access to images in a array-like format.
For each type of images we store:
 - the voxel extracted by the standard mask
 - the non-subsampled non-smoothed images (mask without cerebellum)

The order is read from the file subject_id.csv.

We do that for:
 - non-smoothed images
 - smoothed images (FWMH=10)
 - smoothed images (FWMH=4.71)
 - smoothed images (FWMH=9.42)
 - smoothed images (FWMH=14.13)

Warning: existing files may be deleted

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

# R does not deal well with int64 so we use int32
subjects_id = pandas.io.parsers.read_csv(INPUT_SUBJECTS_ID_FILE, squeeze=True).astype('int32')
subjects_id_index = pandas.Index(subjects_id, name='subject_id')

# Parameers for each image type
IMG_PATH = [
    os.path.join(DATA_PATH, 'VBM', 'gaser_vbm8/'),  # Non-smoothed
    os.path.join(DATA_PATH, 'VBM', 'gaser_vbm8/'),  # Smoothed
    os.path.join(DATA_PATH, 'VBM', 'gaser_vbm8/', 'smooth_sigma=2'),  # Smoothed FWHM=4.71
    os.path.join(DATA_PATH, 'VBM', 'gaser_vbm8/', 'smooth_sigma=4'),  # Smoothed FWHM=9.42
    os.path.join(DATA_PATH, 'VBM', 'gaser_vbm8/', 'smooth_sigma=6')   # Smoothed FWHM=14.13
    ]

IMG_FILENAME_TEMPLATES = [
  'mwp1{subject_id:012}*.nii',    # Non-smoothed
  'smwp1{subject_id:012}*.nii',   # Smoothed
  's2mwp1{subject_id:012}*.nii',  # Smoothed FWHM=4.71
  's4mwp1{subject_id:012}*.nii',  # Smoothed FWHM=9.42
  's6mwp1{subject_id:012}*.nii'   # Smoothed FWHM=14.13
  ]

MASK_PATH = [
  os.path.join(DATA_PATH, 'mask', 'mask.nii'),
  os.path.join(DATA_PATH, 'mask', 'mask.nii'),
  os.path.join(DATA_PATH, 'mask', 'mask.nii'),
  os.path.join(DATA_PATH, 'mask', 'mask.nii'),
  os.path.join(DATA_PATH, 'mask', 'mask.nii')
]

MASK_WITHOUT_CEREBELLUM_PATH = [
  os.path.join(DATA_PATH, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii'),
  os.path.join(DATA_PATH, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii'),
  os.path.join(DATA_PATH, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii'),
  os.path.join(DATA_PATH, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii'),
  os.path.join(DATA_PATH, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii')
]

OUTPUT_FILES = [
  os.path.join(DATA_PATH, 'non_smoothed_images.hdf5'),
  os.path.join(DATA_PATH, 'smoothed_images.hdf5'),
  os.path.join(DATA_PATH, 'smoothed_images_sigma=2.hdf5'),
  os.path.join(DATA_PATH, 'smoothed_images_sigma=4.hdf5'),
  os.path.join(DATA_PATH, 'smoothed_images_sigma=6.hdf5')
  ]

PARAM = zip(IMG_PATH, IMG_FILENAME_TEMPLATES,
            MASK_PATH, MASK_WITHOUT_CEREBELLUM_PATH,
            OUTPUT_FILES)

#
# Do it
#
for (img_path, img_filenames_template, mask_file, mask_cerebellum_file, output_file) in PARAM:
    # Check if output file exists
    if os.path.exists(output_file):
        print "Warning: continuing will erase %s" % output_file
        a = raw_input("Are you sure [Y/n]?")
        if a != 'Y':
            continue

    print "Creating file %s" % output_file
    h5file = tables.openFile(output_file, mode="w")

    # Find images
    files = bmi_utils.find_images(subjects_id, img_filenames_template, img_path)
    print "Found", len(files), "images"

    # Read & store smoothed data with standard mask
    babel_mask  = nibabel.load(mask_file)
    images = bmi_utils.read_images_with_mask(files, babel_mask)
    print "Images with standard mask read"
    bmi_utils.store_images_and_mask(h5file, images, babel_mask, group_name="standard_mask")
    print "Images with standard mask dumped"
    del images

    # Read & store smoothed data without cerebellum mask
    babel_mask_without_cerebellum  = nibabel.load(mask_cerebellum_file)
    images_without_cerebellum = bmi_utils.read_images_with_mask(files, babel_mask_without_cerebellum)
    print "Images without cerebellum read"
    bmi_utils.store_images_and_mask(h5file, images_without_cerebellum, babel_mask_without_cerebellum, group_name="mask_without_cerebellum")
    print "Images without cerebellum dumped"
    del images_without_cerebellum

    h5file.close()