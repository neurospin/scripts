# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:05:57 2016

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import re
import glob

GROUP_MAP = {1: 1, '2a': 2, '2b': 3, 3 : 0}
GENDER_MAP = {'f': 0, 'm': 1}



BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_AUSZ/documents/Tableau_IRM_AUSZ_GM_03_11.xlsx'
INPUT_FS = '/neurospin/brainomics/2016_AUSZ/preproc_FS/freesurfer_assembled_data_fsaverage'

OUTPUT_CSV = os.path.join(BASE_PATH,"results","Freesurfer","population.csv")


#ICAAR + EUGEI population file
##############################################################################
# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME,header = (1))
assert  clinic.shape == (129, 10)


# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-10]
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject =os.path.basename(path)[:-10]
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["image", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (120, 3)

# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="IRM.1")
pop = clinic[["image", "group_inclusion", "group_outcome", "sex", "age","cohort", "mri_path_lh",  "mri_path_rh"]]
assert  pop.shape == (75, 8)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['cohort.num'] = pop["cohort"].map(COHORT_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR_EUGEI, encoding='utf-8' ) 
##############################################################################



#ICAAR only population file
##############################################################################
# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (76, 14)


# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-7]
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject =os.path.basename(path)[:-7]
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["image", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (76, 3)

# intersect with subject with image
clinic = clinic[clinic.cohort!='eugei']
clinic = clinic.merge(input_subjects_fs, on="image")
pop = clinic[["image","group_inclusion", "group_outcome", "sex", "age","cohort", "mri_path_lh",  "mri_path_rh"]]
assert  pop.shape == (55, 8)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['cohort.num'] = pop["cohort"].map(COHORT_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR, encoding='utf-8' ) 
##############################################################################


