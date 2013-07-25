'''A module to ease access to the subdepression database.
   The basic idea is to store images in a HDF5 file for fast access.
'''

import os.path
import collections
import tables, pandas

# Some constants
CLINIC_DIRECTORY_NAME = 'clinic'
CLINIC_FILE_NAME      = 'fichier_groups_119sub_461ctl_version2.csv'
DATA_DIRECTORY_NAME  =  'data_normalized_segmented'
SEGMENTED_IMAGES_DIRECTORY_NAME  ='new_segment_spm8'
MASK_FILE_NAME        = 'mask.img'
IMAGE_SIZE            = (121, 145, 121)

#
# List the values of the categorical variables and the associated numeric values.
# The first value is the reference level in dummy coding
# Using OrderedDict allow deterministic order which is important for dummy coding.
#

# The values are arbitrary
GroupMap      = collections.OrderedDict((('control', 0), ('sub', 1)))
# The values are arbitrary
GenderMap     = collections.OrderedDict((('Male', 0), ('Female', 1)))
# The values are arbitrary (there are some missing values)
HandednessMap = collections.OrderedDict((
                  ('NA',          0),
                  ('Right',       1),
                  ('Left',        2),
                  ('Both-handed', 3)))
# The values are in the Center column
# The names in the ImagingCenterCity column
CityMap       = collections.OrderedDict((
                  ('LONDON',     1),
                  ('NOTTINGHAM', 2),
                  ('DUBLIN',     3),
                  ('BERLIN',     4),
                  ('HAMBURG',    5),
                  ('MANNHEIM',   6),
                  ('PARIS',      7),
                  ('DRESDEN',    8)))
# The values are in the Scanner column
# The names in the Scanner_Type column
ScannerMap    = collections.OrderedDict((
                  ('GE',      1),
                  ('Philips', 2),
                  ('Siemens', 3)))

# Categorical variables mappings
REGRESSOR_MAPPINGS = {
"group_sub_ctl":     GroupMap,
"Gender":            GenderMap,
"Scanner_Type":      ScannerMap,
"ImagingCentreCity": CityMap
}

def get_clinic_dir_path(base_path, test_exist=True):
    '''Returns the name of the clinic dir'''
    clinic_dir_path = os.path.join(base_path, CLINIC_DIRECTORY_NAME)
    if not test_exist or os.path.exists(clinic_dir_path):
        return clinic_dir_path
    else:
        raise Exception("Directory %s does not seem to exist" % clinic_dir_path)

def get_clinic_file_path(base_path, test_exist=True):
    '''Returns the name of the clinic CSV file'''
    csv_file_path = os.path.join(get_clinic_dir_path(base_path, test_exist=False),
                                 CLINIC_FILE_NAME)
    if not test_exist or os.path.exists(csv_file_path):
        return csv_file_path
    else:
        raise Exception("File %s does not seem to exist" % csv_file_path)

def get_images_dir_path(base_path, test_exist=True):
    '''Returns the name of the image dir'''
    image_dir_path = os.path.join(base_path, DATA_DIRECTORY_NAME, SEGMENTED_IMAGES_DIRECTORY_NAME)
    if not test_exist or os.path.exists(image_dir_path):
        return image_dir_path
    else:
        raise Exception("Directory %s does not seem to exist" % image_dir_path)

def get_mask_file_path(base_path, test_exist=True):
    '''Returns the name of the mask file'''
    mask_file_path = os.path.join(base_path,
                                  DATA_DIRECTORY_NAME,
                                  'sub_vs_ctl_article',
                                  MASK_FILE_NAME)
    if not test_exist or os.path.exists(mask_file_path):
        return mask_file_path
    else:
        raise Exception("File %s does not seem to exist" % mask_file_path)

def write_images(h5file, X):
    '''Write grey matter masked images'''
    # X
    atom = tables.Atom.from_dtype(X.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'masked_images', atom, X.shape, filters=filters)
    ds[:] = X

def get_images(h5file):
    '''Return images from the HDF5 file'''
    return h5file.root.masked_images

def read_clinic_file(path):
    '''Read the clinic filename and perform basic conversions.
       Return a dataframe.'''
    # The first column is the line number so we can skip it
    df = pandas.io.parsers.read_csv(path, index_col=0)
    return df