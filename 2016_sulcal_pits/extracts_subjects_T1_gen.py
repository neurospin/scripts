"""
Created  11 17 2015

@author yl247235
"""

import pandas as pd
import numpy as np
import re, os, glob

import json
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

path = '/neurospin/imagen/workspace/cati/BVdatabase/'
centres = ['Berlin','Dresden', 'Dublin', 'Hamburg', 'London', 'Mannheim', 'Nottingham', 'Paris']
subjects_nifti_BL = []
for centre in centres:
    path_c = path+centre+'/'
    for subject in os.listdir(path_c):
        path_s =  path_c+subject+'/'
        if os.path.isdir(path_s):
            path_nifti = path_s + 't1mri/BL/'+subject+'.nii.gz'
            if os.path.isfile(path_nifti):
                subjects_nifti_BL.append(subject)

subjects_nifti_BL = [u'%012d' % int(i) for i in subjects_nifti_BL]
labels = [subject for subject in subject_ids if subject in subjects_nifti_BL]
labels = [int(i) for i in labels]
labels = np.sort(labels)
labels = [u'%012d' % int(i) for i in labels]
path_saved = '/neurospin/brainomics/imagen_central/'
filename_sav = 'subjects_T1_gen_Brainvisa.txt'
thefile = open(path_saved+ filename_sav, 'w')
for item in labels:
  thefile.write("%s " % item)
thefile.close()

"""path_source_BL = '/neurospin/imagen/BL/processed/nifti/'
T1_sub_dir = 'SessionA/ADNI_MPRAGE/'
subjects_BL = [subject for subject in os.listdir(path_source_BL) if os.path.isdir(os.path.join(path_source_BL,subject))]

subjects_BL_real = []
for subject in subjects_BL:
    for filename in glob.glob(os.path.join(path_source_BL+subject+'/'+T1_sub_dir,'*nii.gz')):
        if os.path.basename(filename) == 'o'+os.path.basename(filename)[1:len(os.path.basename(filename))]:
            pass
        elif os.path.basename(filename) == 'co' +os.path.basename(filename)[2:len(os.path.basename(filename))]:
            pass
        else:
            #print filename
            subjects_BL_real.append(subject)

IID_T1_QC_gen_real = [subject for subject in subject_ids if subject in subjects_BL_real] """
