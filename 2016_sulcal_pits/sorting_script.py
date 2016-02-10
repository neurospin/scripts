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
            path_fr = path_s + 'surface/'+subject+'_Rwhite_rectangular_flat.gii'
            path_fl = path_s + 'surface/'+subject+'_Lwhite_rectangular_flat.gii'
            if not os.path.isfile(path_fr) and os.path.isfile(path_nifti):
                paths_R.append(path_fr)
                subjects_to_redo_R.append(subject)
            if not os.path.isfile(path_fl) and os.path.isfile(path_nifti):
                paths_L.append(path_fl)
                subjects_to_redo_L.append(subject)

path_saved = '/neurospin/brainomics/2016_sulcal_pits/'
filename_sav = 'subject_redo_R.txt'

# Create the file fist else not all information are written
thefile = open(path_saved+ filename_sav, 'wb')
thefile.close

thefile = open(path_saved+ filename_sav, 'wb')
for item in subjects_to_redo_R:
  thefile.write("%s " % item)
thefile.close
