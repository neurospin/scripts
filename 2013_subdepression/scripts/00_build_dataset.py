# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

This script concatenates all the data in a huge HDF5 file.
This is used mainly for fast local access.

"""

# Standard library modules
import os, sys, argparse
import math
# Numpy and friends
import numpy, pandas, patsy
# For writing HDF5 files
import tables
# For loading images
import nibabel

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

def dump_in_hdf5(DB_PATH, outfilename, title):
    # TODO: use the number of subjects as the number of records of the table
    images_dir = data_api.get_images_dir_path(args.DB_PATH)
    clinic_filename = data_api.get_clinic_file_path(args.DB_PATH)
    # Open the output file
    h5file = tables.openFile(outfilename, mode = "w", title = title)
    
    # Open the clinic file
    csv_fd = open(clinic_filename)
    data = pandas.io.parsers.read_csv(csv_fd)
    #numpy_data = data.to_records()
    n_subjects = data.shape[0]
    
    # Load mask
    mask_filename = data_api.get_mask_file_path(args.DB_PATH)
    print "Loading mask {mask_filename}".format(mask_filename=mask_filename)
    babel_mask = nibabel.load(mask_filename)
    mask = babel_mask.get_data()
    binary_mask = mask!=0
    useful_voxels = numpy.ravel_multi_index(numpy.where(binary_mask), mask.shape)
    n_useful_voxels = len(useful_voxels)
    print "Mask loaded ({n_useful_voxels} useful voxels per image)".format(n_useful_voxels=n_useful_voxels)
    
    # Load grey matter images (X), apply mask and concatenate them
    print "Loading {n_images} images, apply mask and flatten".format(n_images=n_subjects)
    image_filenames = [os.path.join(images_dir, 'mwc1' + filename) for filename in data.Images]
    X = numpy.zeros((n_subjects, n_useful_voxels))
    for (index, filename) in enumerate(image_filenames):
        # Load (as numpy array)
        image = nibabel.load(filename).get_data()
        # Apply mask (returns a flat image)
        masked_image = image[binary_mask]
        # Store in X
        X[index, :] = masked_image
    
    # Load regressors, dummy code them and concatenate them
    Y_data = {}
    for column, mapper in zip(data_api.REGRESSORS, data_api.REGRESSOR_MAPS):
        if mapper:
            for level in data[column].unique():
                column_name = '{cat}_{value}'.format(cat=column, value=level)
                column_serie = (data[column] == level).astype(numpy.float64)
                Y_data[column_name] = column_serie
        else:
            Y_data[column] = data[column]
    Y = pandas.DataFrame(Y_data).values # Very inefficient

    # Create groups and table
    # X
    atom = tables.Atom.from_dtype(X.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = h5file.createCArray(h5file.root, 'X', atom, X.shape, filters=filters)
    ds[:] = X
    # Y
    atom = tables.Atom.from_dtype(Y.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = h5file.createCArray(h5file.root, 'Y', atom, Y.shape, filters=filters)
    ds[:] = Y
    # Subject ids
    subject_id = data['Subject'].values
    atom = tables.Atom.from_dtype(subject_id.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = h5file.createCArray(h5file.root, 'subject_id', atom, subject_id.shape, filters=filters)
    ds[:] = subject_id
    h5file.close()
    
    return X, Y

if __name__ == '__main__':
    # Parse CLI
    parser = argparse.ArgumentParser(description='''Concatenates all the data
                                                    in a huge HDF5 file''')
    
    parser.add_argument('DB_PATH',
      help='Path of the database')
    
    parser.add_argument('outfilename',
      type=str,
      help='Write to outfilename')

    parser.add_argument('--title',
      type=str, default='imagen_subdepression',
      help='Title')

    args = parser.parse_args()
    X, Y = dump_in_hdf5(**vars(args))
