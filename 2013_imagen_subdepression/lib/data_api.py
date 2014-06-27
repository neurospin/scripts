'''A module to ease access to the subdepression database.
   The basic idea is to store images in a HDF5 file for fast access.
   TODO: what about categorical variables that are not strings (integers)?
'''

import os.path
import collections
import tables, pandas

# Some constants
DATA_DIRECTORY_NAME  =  'data'
SEGMENTED_IMAGES_DIRECTORY_NAME  = os.path.join('VBM', 'new_segment_spm8')
MASK_FILE_NAME        = 'mask.img'
IMAGE_SIZE            = (121, 145, 121)
CLINIC_DIRECTORY_NAME = 'clinic'
CLINIC_FILE_NAME      = 'fichier_groups_119sub_461ctl_version2.csv'

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

QualityControlMap = collections.OrderedDict((
                  ('A', 1),
                  ('B', 2)))
                  
TemplateMap =  collections.OrderedDict((
                  ('no',  0),
                  ('yes', 1)))

# Categorical variables mappings
REGRESSOR_MAPPINGS = {
"group_sub_ctl":     GroupMap,
"Gender":            GenderMap,
"Gender de Feuil2":  GenderMap,
"Scanner_Type":      ScannerMap,
"ImagingCentreCity": CityMap,
"Handedness":        HandednessMap,
"quality_control":   QualityControlMap,
"template":          TemplateMap
}

# Some columns are redundant so I remove them:
#  - Center is the numerical representation of ImagingCentreCity
#  - Scanner is the numerical representation of Scanner_Type
# Also the image field is special
COLUMNS = ['Subject', 'group_sub_ctl', 'Gender', 'pds', 'Age',
           'ImagingCentreCity', 'Scanner_Type',
           'quality_control', 'template',
           'vol_GM', 'vol_WM', 'vol_CSF', 'TIV', 'GM_on_TIV', 'WM_on_TIV', 'CSF_on_TIV', 'VSF', 'Handedness',
           'IQ', 'tristesse', 'irritabilite', 'anhedonie', 'total_symptoms_dep']

def get_clinic_dir_path(base_path, test_exist=True):
    '''Returns the name of the clinic dir'''
    clinic_dir_path = os.path.join(base_path, DATA_DIRECTORY_NAME, CLINIC_DIRECTORY_NAME)
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

def write_images(h5file, X, name='masked_images'):
    '''Write grey matter masked images'''
    # X
    atom = tables.Atom.from_dtype(X.dtype)
    filters = tables.Filters(complib='zlib', complevel=5)
    ds = h5file.createCArray(h5file.root, name, atom, X.shape, filters=filters)
    ds[:] = X

def get_images(h5file, name='masked_images'):
    '''Return images from the HDF5 file'''
    return h5file.getNode(h5file.root, name)

def read_clinic_file(path):
    '''Read the clinic filename and perform basic conversions.
       Return a dataframe.'''
    # The first column is the line number so we can skip it
    df = pandas.io.parsers.read_csv(path, index_col=0)
    df = df[COLUMNS]
    return df