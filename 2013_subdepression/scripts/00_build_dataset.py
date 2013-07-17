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
import numpy, pandas
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
    mask_affine = babel_mask.get_affine()
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
    
    # Map regressors, values
    n_cols = len(data_api.REGRESSORS)
    Y = numpy.zeros((n_subjects, n_cols))
    for col_index, (column, mapping) in enumerate(zip(data_api.REGRESSORS, data_api.REGRESSOR_MAPPINGS)):
        if mapping:
            # Map values
            print "Mapping categorical variable %s at col %i" % (column, col_index)
            Y[:, col_index] = numpy.array([mapping[l] for l in data[column]])
        else:
            print "Putting ordinal variable %s at col %i" % (column, col_index)
            Y[:, col_index] = data[column]

    # Store data
    subject_id = data['Subject'].values
    data_api.write_data(h5file, X, Y, subject_id, mask, mask_affine)
    
    # Dummy coding
    Y_dummy = data_api.dummy_coding(Y)
    data_api.write_dummy(h5file, Y_dummy)

    h5file.close()

    return X, Y, Y_dummy

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
    X, Y, Y_dummy = dump_in_hdf5(**vars(args))
