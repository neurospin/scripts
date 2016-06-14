"""

@author: yl247234
Copyrignt : CEA NeuroSpin - 2015
"""

import glob
import os
import shutil
import re
import numpy as np


path = '/neurospin/imagen/workspace/cati/BVdatabase/'

centres = ['Berlin','Dresden', 'Dublin', 'Hamburg', 'London', 'Mannheim', 'Nottingham', 'Paris']

subjects_to_redo_R = []
paths_R = []
subjects_to_redo_L = []
paths_L = []
for centre in centres:
    path_c = path+centre+'/'
    for subject in os.listdir(path_c):
        path_s =  path_c+subject+'/'
        if os.path.isdir(path_s):
            path_nifti = path_s + 't1mri/BL/'+subject+'.nii.gz'
            path_fr = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Rwhite_pits_smoothed_on_atlas.gii'
            path_fl = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Lwhite_pits_smoothed_on_atlas.gii'
            path_sl = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Lwhite_spherical.gii'
            path_sr = path_s + 't1mri/BL/default_analysis/segmentation/mesh/surface_analysis/'+subject+'_Rwhite_spherical.gii'
            if not os.path.isfile(path_fr) and os.path.isfile(path_sr):
                paths_R.append(path_fr)
                subjects_to_redo_R.append(subject)
            if not os.path.isfile(path_fl) and os.path.isfile(path_sl):
                paths_L.append(path_fl)
                subjects_to_redo_L.append(subject)


path_saved = '/neurospin/imagen/workspace/cati/'
filename_sav = 'subject_redo_smooth_pits_R.txt'
thefile = open(path_saved+ filename_sav, 'w')
for item in subjects_to_redo_R:
  thefile.write("%s " % item)
thefile.close()

"""import json
from genibabel import imagen_subject_ids
# Consider subjects for who we have neuroimaging and genetic data
# To fix genibabel should offer a iid function -direct request to server
login = json.load(open(os.environ['KEYPASS']))['login']
passwd = json.load(open(os.environ['KEYPASS']))['passwd']
#Set the data set of interest ("QC_Genetics", "QC_Methylation" or "QC_Expression")
data_set = "QC_Genetics"
# Since login and password are not passed, they will be requested interactily
subject_ids = imagen_subject_ids(data_of_interest=data_set, login=login,
                                 password=passwd)

subjects_to_really_redo_R = set(subjects_to_redo_R)-(set(subjects_to_redo_R)-set(subject_ids)) #69 (87)

subjects_to_really_redo_L = set(subjects_to_redo_L)-(set(subjects_to_redo_L)-set(subject_ids)) #70 (93)"""


"""path_saved = '/neurospin/brainomics/2016_sulcal_pits/'
filename_sav = 'subject_redo_R.txt'



thefile = open(path_saved+ filename_sav, 'w')
for item in subjects_to_redo_R:
  thefile.write("%s " % item)
thefile.close()"""
