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
    n_cols = sum(map(data_api.n_columns, data_api.REGRESSOR_VALUES))
    print "Expanding categorical variables yields %i columns" % n_cols

    Y = numpy.zeros((n_subjects, n_cols))
    col_index = 0
    for column, values in zip(data_api.REGRESSORS, data_api.REGRESSOR_VALUES):
        if values:
            if len(values) == 2:
                # Map all the values
                print "Putting binary variable %s at col %i (%s)" % (column, col_index, list(enumerate(values)))
                Y[:, col_index] = numpy.array([values.index(l) for l in data[column]])
                col_index += 1
            else:
                for level in values:
                    column_name = '{cat}_{value}'.format(cat=column, value=level)
                    print "Putting %s at col %i" % (column_name, col_index)
                    column_serie = (data[column] == level).astype(numpy.float64)
                    Y[:, col_index] = column_serie
                    col_index += 1
        else:
            print "Putting ordinal variable %s at col %i" % (column, col_index)
            Y[:, col_index] = data[column]
            col_index += 1

    # Store data
    subject_id = data['Subject'].values
    data_api.write_data(h5file, X, Y, subject_id, mask)

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
