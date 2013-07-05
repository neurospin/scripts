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
REGRESSORS            = ["group_sub_ctl", "Age", "Gender", "VSF", "Scanner_Type"]

# FIELDS = "","Subject","group_sub_ctl","Gender","pds","Age","Center","ImagingCentreCity","Scanner","Scanner_Type","Images","quality_control","template","vol_GM","vol_WM","vol_CSF","TIV","GM_on_TIV","WM_on_TIV","CSF_on_TIV","VSF","Handedness","IQ","tristesse","irritabilite","anhedonie","total_symptoms_dep"


# The values are arbitrary
GroupMap       = {'control': 0, 'sub': 1}
GroupEnum      = tables.misc.enum.Enum(GroupMap)
# The values are arbitrary
GenderMap      = {'Male': 0, 'Female': 1}
GenderEnum     = tables.misc.enum.Enum(GenderMap)
# The values are arbitrary (there are some missing values)
HandednessEnum = tables.misc.enum.Enum({
                'NA'          : 0,
                'Right'       : 1,
                'Left'        : 2,
                'Both-handed' : 3
                })
# The values are in the Center column
# The names in the ImagingCenterCity column
CityEnum       = tables.misc.enum.Enum({
                'LONDON'     : 1,
                'NOTTINGHAM' : 2,
                'DUBLIN'     : 3,
                'BERLIN'     : 4,
                'HAMBURG'    : 5,
                'MANNHEIM'   : 6,
                'PARIS'      : 7,
                'DRESDEN'    : 8
                })
# The values are in the Scanner column
# The names in the Scanner_Type column
ScannerMap     = {
                'GE'      : 1,
                'Philips' : 2,
                'Siemens' : 3
                 }
ScannerEnum    = tables.misc.enum.Enum(ScannerMap)

REGRESSOR_MAPS = [GroupMap, None, GenderMap, None, ScannerMap]

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
