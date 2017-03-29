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

GROUP_MAP = {'N': 0, 'Y': 1}
GENDER_MAP = {'F': 0, 'H': 1}
COHORT_MAP = {'eugei': 0, 'icaar': 1}


BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_icaar-eugei/documents/icaar_eugei_images_correspondanceV2 CMLF.xls'
INPUT_FS = '/neurospin/brainomics/2016_icaar-eugei/preproc_FS/freesurfer_assembled_data_fsaverage'

QC_FS = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/Freesurfer/FS_QC.xlsx"


OUTPUT_CSV_ICAAR_EUGEI = os.path.join(BASE_PATH,"2017_icaar_eugei","Freesurfer","ICAAR+EUGEI","population.csv")


##ICAAR + EUGEI population file
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
clinic = clinic.merge(input_subjects_fs, on="image")
pop = clinic[["image", "group_inclusion", "group_outcome", "sex", "age","cohort", "mri_path_lh",  "mri_path_rh"]]
assert  pop.shape == (75, 8)

qc = pd.read_excel(QC_FS)
images_to_exclude = qc["IMAGES"][qc["DECISION"]==1].reset_index()
for i in range(len(images_to_exclude)):
    pop = pop[pop.image != images_to_exclude.IMAGES[i]]

assert  pop.shape == (70, 8)
pop = pop.reset_index()



# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['cohort.num'] = pop["cohort"].map(COHORT_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR_EUGEI, encoding='utf-8' ) 
###############################################################################




