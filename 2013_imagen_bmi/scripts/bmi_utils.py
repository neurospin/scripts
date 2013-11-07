# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:13:26 2013

@author: md238665
"""

import os, fnmatch
import numpy
import nibabel
import tables


def find_images(subjects_id, pattern, directory):
    filenames = []
    img_dir_files = os.listdir(directory)
    for (index, subject_index) in enumerate(subjects_id):
        # Find filename
        file_pattern = pattern.format(subject_id=subject_index)
        filename = fnmatch.filter(img_dir_files, file_pattern)
        if len(filename) != 1:
            raise Exception
        else:
            filename = os.path.join(directory, filename[0])
            if os.path.exists(filename):
                filenames.append(filename)
            else:
                raise Exception
    return filenames


def read_images_with_mask(list_filenames, babel_mask):
    mask        = babel_mask.get_data()
    binary_mask = mask!=0
    useful_voxels = numpy.ravel_multi_index(numpy.where(binary_mask), mask.shape)
    n_useful_voxels = len(useful_voxels)
    n_subjects = len(list_filenames)
    # Allocate data
    masked_images = numpy.empty((n_subjects, n_useful_voxels))
    for (index, filename) in enumerate(list_filenames):
        # Load (as numpy array)
        image = nibabel.load(filename).get_data()
        # Apply mask (returns a flat image)
        masked_image = image[binary_mask]
        # Store in array
        masked_images[index, :] = masked_image
    return masked_images


def store_images_and_mask(h5file, images, babel_mask, group_name, covar=None, group_title=''):
    # Create group
    full_group_name = '/' + group_name
    h5file.createGroup(h5file.root, group_name, title=group_title)
    filters = tables.Filters(complib='zlib', complevel=5)
    # Store images
    atom = tables.Atom.from_dtype(images.dtype)
    ds = h5file.createCArray(full_group_name, 'masked_images', atom, images.shape, filters=filters)
    ds[:] = images
    # Store mask
    mask = babel_mask.get_data()
    atom = tables.Atom.from_dtype(mask.dtype)
    ds = h5file.createCArray(full_group_name, 'mask', atom, mask.shape, filters=filters)
    ds[:] = mask
    # Store covar
    if covar is not None:
        atom = tables.Atom.from_dtype(images.dtype)
        ds = h5file.createCArray(full_group_name, 'covar', atom, covar.shape, filters=filters)
        ds[:] = covar


def store_array(h5file, array, name):
    if not isinstance(array, numpy.ndarray):
        array = numpy.array(array)
    filters = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Atom.from_dtype(array.dtype)
    ds = h5file.createCArray(h5file.root, name, atom, array.shape, filters=filters)
    ds[:] = array


def read_array(h5file, name):
    # It seems that it is better to allocate array before
    # Otherwise images take ages to load
    hdf_array = h5file.getNode(name)
    array = numpy.empty(hdf_array.shape, hdf_array.dtype)
    array = hdf_array[:]
    return array