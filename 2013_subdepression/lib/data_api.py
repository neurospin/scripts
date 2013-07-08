'''A module to ease access to the subdepression database.'''

# TODO: column position

import os.path
import tables, numpy

CLINIC_DIRECTORY_NAME = 'clinic'
CLINIC_FILE_NAME      = 'fichier_groups_119sub_461ctl_version2.csv'
DATA_DIRECTORY_NAME  =  'data_normalized_segmented'
SEGMENTED_IMAGES_DIRECTORY_NAME  ='new_segment_spm8'
MASK_FILE_NAME        = 'mask.img'
IMAGE_SIZE            = (121, 145, 121)

#
# List the values of the categorical variables.
# Using lists allow deterministic order.
#

# Values of the group column
GroupValues = ['control', 'sub']
# Values of the gender column
GenderValues = ['Male', 'Female']
# Values of the handedness column
HandednessValues = ['NA',
                    'Right',
                    'Left',
                    'Both-handed']
# Values of the Scanner_Type column
ScannerValues = ['GE',
                 'Philips',
                 'Siemens']

REGRESSORS       = ["group_sub_ctl", "Age", "Gender", "VSF", "Scanner_Type"]
REGRESSOR_VALUES = [GroupValues, None, GenderValues, None, ScannerValues]

# Determine the number of columns for a given categorical :
#  - if the values is None -> 1 column
#  - if len(values) is 2 -> 1 column
#  - else len(values) columns
def n_columns(values):
    if values == None:
        return 1
    if len(values) == 2:
        return 1
    else:
        return len(values)

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

def write_data(h5file, X, Y, subject_id, mask):
    '''Write basic representation of the dataset (X, Y, mask ad subject_id)'''
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
    atom = tables.Atom.from_dtype(subject_id.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = h5file.createCArray(h5file.root, 'subject_id', atom, subject_id.shape, filters=filters)
    ds[:] = subject_id
    # Mask
    atom = tables.Atom.from_dtype(mask.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = h5file.createCArray(h5file.root, 'mask', atom, mask.shape, filters=filters)
    ds[:] = mask

def get_data(h5file):
    '''Return data from the HDF5 file'''
    return (h5file.root.X, h5file.root.Y, h5file.root.mask)
