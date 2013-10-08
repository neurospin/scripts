# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:47:44 2013

@author: md238665

This script subsamples the Gaser-segmented images and the mask and dump them in a HDF5 file for faster access.

The segmentation procedure is borrowed from Vincent Frouin subsampling scripts.
It uses a recent version of nilearn.
The git repository can be cloned at https://github.com/nilearn/nilearn.git

"""

import os, fnmatch
import numpy, pandas
import tables
import nibabel
from nilearn.image.resampling import resample_img

# Input
BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'
DATA_DIR=os.path.join(BASE_DIR, 'data')
CLINIC_DIR=os.path.join(DATA_DIR, 'clinic');
IMAGES_DIR=os.path.join(DATA_DIR, 'VBM', 'gaser_vbm8')
INIMG_FILENAME_TEMPLATE='smwp1{subject_id:012}*.nii'

# Original mask
MASK_FILE=os.path.join(DATA_DIR, 'mask_without_cerebellum', 'mask_without_cerebellum_7.nii')

# Images will be shaped as this images
TARGET_FILE='/neurospin/brainomics/neuroimaging_ressources/atlases/HarvardOxford/HarvardOxford-LR-cort-333mm.nii.gz'

# Output files
OUT_DIR=os.path.join(DATA_DIR, 'reduced_images')
OUTIMG_FILENAME_TEMPLATE='rsmwp1{subject_id:012}*.nii'
RMASK_FILE=os.path.join(OUT_DIR, 'rmask.nii')
OUT_HDF5_FILE=os.path.join(OUT_DIR, 'cache.hdf5')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def resample(input_image, target_affine, target_shape, interpolation='continuous'):
    outim = resample_img(input_image,
                         target_affine=target_affine, target_shape=target_shape,
                         interpolation=interpolation)
    return outim

# Open the clinic file to get subject's ID
df = pandas.io.parsers.read_csv(os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv'))
subject_indices = df.Subjects
n_images = df.shape[0]

# Open target image
target = nibabel.load(TARGET_FILE)
target_shape = target.shape
target_affine = target.get_affine()

# Open mask
babel_mask  = nibabel.load(MASK_FILE)

# Resize mask & save it
babel_rmask = resample(babel_mask, target_affine, target_shape, interpolation='nearest')
nibabel.save(babel_rmask, RMASK_FILE)
rmask        = babel_rmask.get_data()
binary_rmask = rmask!=0
useful_voxels = numpy.ravel_multi_index(numpy.where(binary_rmask), rmask.shape)
n_useful_voxels = len(useful_voxels)
print "Mask reduced: {n} true voxels".format(n=n_useful_voxels)

# Read images in the same order than subjects, resample & dump
masked_images = numpy.zeros((n_images, n_useful_voxels))
images_dir_files = os.listdir(IMAGES_DIR)
for (index, subject_index) in enumerate(subject_indices):
    # Find filename
    pattern = INIMG_FILENAME_TEMPLATE.format(subject_id=subject_index)
    filename = fnmatch.filter(images_dir_files, pattern)
    if len(filename) != 1:
        raise Exception
    else:
        full_infilename = os.path.join(IMAGES_DIR, filename[0])
    # Generate output name
    full_outfilename = os.path.join(OUT_DIR, 'r'+filename[0])
    print "Reducing {in_} to {out}".format(in_=full_infilename, out=full_outfilename)
    # Resample and save
    input_image = nibabel.load(full_infilename)
    out_image = resample(input_image, target_affine, target_shape)
    nibabel.save(out_image, full_outfilename)
    # Apply mask (returns a flat image)
    masked_image = out_image.get_data()[binary_rmask]
    # Store in array
    masked_images[index, :] = masked_image
# Open the HDF5 file
h5file = tables.openFile(OUT_HDF5_FILE, mode = "w", title = 'reduced_images')
atom = tables.Atom.from_dtype(masked_images.dtype)
filters = tables.Filters(complib='zlib', complevel=5)
ds = h5file.createCArray(h5file.root, 'masked_images', atom, masked_images.shape, filters=filters)
ds[:] = masked_images
h5file.close()
print "Images reduced and dumped"
