'''A module to ease access to the subdepression database.
   The basic idea is to store everything in a HDF5 file.
   The database is first compressed in the HDF5 file:
    - X: concatenated masked images (size (n_subjects, n_useful_voxels))
    - Y: concatenated regressors (the first column is the group) (size (n_subjects, n_regressors))
    - mask: the mask
   Dummy codig can be peformed and stored in Y_dummy.
   TODO: add indicator variables coding?
   TODO: use named array
'''

import os.path
import collections
import tables, numpy

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

# Variables name and mappings
# None is used to indicate ordinal variables
REGRESSORS         = ["group_sub_ctl", "Age", "Gender", "VSF", "Scanner_Type"]
REGRESSOR_MAPPINGS = [GroupMap, None, GenderMap, None, ScannerMap]

def n_dummy_columns(mapping):
    '''Determine the number of columns for dummy coding of a given categorical variable mapping:
         - if the values is None -> 1 column
         - else len(values)-1 columns'''
    if mapping == None:
        return 1
    else:
        return len(mapping) - 1

def dummy_coding(Y):
    '''Return a dummy coded Y matrix'''
    n_subjects = Y.shape[0]
    variables = REGRESSORS
    mappings  = REGRESSOR_MAPPINGS
    n_cols = sum(map(n_dummy_columns, mappings))
    print "Expanding categorical variables yields %i columns" % n_cols

    Y_dummy = numpy.zeros((n_subjects, n_cols))
    col_index = 0
    dummy_col_index = 0
    for variable, mapping in zip(variables, mappings):
        if mapping:
            categorical_values = mapping.keys()
            categorical_ref_value = categorical_values.pop(0)
            numerical_values = mapping.values();
            numerical_values = mapping.values();
            numerical_ref_value = numerical_values.pop(0)
            print "Reference value for %s is %s (%d)" % (variable, categorical_ref_value, numerical_ref_value)
            for level_index, level in enumerate(numerical_values):
                dummy_var = '{cat}_d{value}'.format(cat=variable, value=level_index)
                print "Putting %s at col %i" % (dummy_var, dummy_col_index)
                column_serie = (Y[:, col_index] == level).astype(numpy.float64)
                Y_dummy[:, dummy_col_index] = column_serie
                dummy_col_index += 1
            col_index += 1
        else:
            print "Putting ordinal variable %s at col %i" % (variable, dummy_col_index)
            Y_dummy[:, dummy_col_index] = Y[:, col_index]
            dummy_col_index += 1
            col_index += 1
    return Y_dummy

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

def write_data(h5file, X, Y, subject_id, mask, affine_transform):
    '''Write basic representation of the dataset (X, Y, mask ad subject_id)'''
    # X
    atom = tables.Atom.from_dtype(X.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'X', atom, X.shape, filters=filters)
    ds[:] = X
    # Y
    atom = tables.Atom.from_dtype(Y.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'Y', atom, Y.shape, filters=filters)
    ds[:] = Y
    # Subject ids
    atom = tables.Atom.from_dtype(subject_id.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'subject_id', atom, subject_id.shape, filters=filters)
    ds[:] = subject_id
    # Mask
    atom = tables.Atom.from_dtype(mask.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'mask', atom, mask.shape, filters=filters)
    ds[:] = mask
    # Mask affine transform
    atom = tables.Atom.from_dtype(affine_transform.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'mask_affine_transform', atom, affine_transform.shape, filters=filters)
    ds[:] = affine_transform

def write_dummy(h5file, Y_dummy):
    '''Write dummy variables'''
    atom = tables.Atom.from_dtype(Y_dummy.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, 'Y_dummy', atom, Y_dummy.shape, filters=filters)
    ds[:] = Y_dummy

def get_data(h5file):
    '''Return data from the HDF5 file'''
    return (h5file.root.X, h5file.root.Y, h5file.root.mask, h5file.root.mask_affine_transform)

def get_dummy(h5file):
    return h5file.root.Y_dummy