# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:45:36 2013

@author: Mathieu Dubois (mathieu.dubois@cea.fr)

This script concatenates all the masked images in a huge HDF5 file.
This is used mainly for fast local access.

"""

# Standard library modules
import os, sys, argparse
# Numpy and friends
import numpy, pandas
# For writing HDF5 files
import tables
# For loading images
import nibabel

DEFAULT_DB_PATH="/neurospin/brainomics/2012_imagen_subdepression/"
DEFAULT_OUTPUT="/volatile/imagen_subdepression.hdf5"
DEFAULT_TITLE="imagen_subdepression"

# Local import
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
import data_api

def dump_in_hdf5(db_path, output, title):
    # TODO: use the number of subjects as the number of records of the table
    images_dir = data_api.get_images_dir_path(db_path)
    clinic_filename = data_api.get_clinic_file_path(db_path)
    # Open the output file
    h5file = tables.openFile(output, mode = "w", title = title)
    
    # Open the clinic file
    csv_fd = open(clinic_filename)
    data = pandas.io.parsers.read_csv(csv_fd)
    n_subjects = data.shape[0]
    
    # Load mask
    mask_filename = data_api.get_mask_file_path(db_path)
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
    masked_images = numpy.zeros((n_subjects, n_useful_voxels))
    for (index, filename) in enumerate(image_filenames):
        # Load (as numpy array)
        image = nibabel.load(filename).get_data()
        # Apply mask (returns a flat image)
        masked_image = image[binary_mask]
        # Store in X
        masked_images[index, :] = masked_image

    # Store data
    data_api.write_images(h5file, masked_images)

    h5file.close()

if __name__ == '__main__':
    # Parse CLI
    parser = argparse.ArgumentParser(description='''Load DB images, apply mask and flatten.
      Store the results in a HDF5 file''')
    
    parser.add_argument('--db_path',
      type=str, default=DEFAULT_DB_PATH,
      help='Path of the database (default: %s)'% (DEFAULT_DB_PATH))
    
    parser.add_argument('--output',
      type=str, default=DEFAULT_OUTPUT,
      help='Write to OUTPUT (default: %s)' % (DEFAULT_OUTPUT))

    parser.add_argument('--title',
      type=str, default=DEFAULT_TITLE,
      help='Title (default %s)' % (DEFAULT_TITLE))

    args = parser.parse_args()
    dump_in_hdf5(**vars(args))
