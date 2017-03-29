#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:04:05 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import re
import glob


BASE_PATH = "/neurospin/brainomics/2017_asd_charles/Freesurfer"
INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "doc","amicie_asd.csv")
INPUT_FS = os.path.join(BASE_PATH, "preproc","freesurfer_assembled_data_fsaverage")

OUTPUT_CSV = os.path.join(BASE_PATH,"results","population.csv")


# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME,sep = ";")
assert  clinic.shape == (367, 7)

# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-7]
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-7]
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["PatientID", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (451, 3)


# intersect with subject with image
pop = clinic.merge(input_subjects_fs, on="PatientID")
assert  pop.shape == (367,9)




SEX_MAP = {'m': 0, 'f': 1}
pop['sex_num'] = pop["sexe"].map(SEX_MAP)

DX_MAP = {2: 0, 1: 1}
pop['dx_num'] = pop["diag1AUT_2TEM"].map(DX_MAP)

SITE_MAP = {'CRE': 1, 'USM': 2, 'IU':3,'NYU':4,'OLI':5,'BNI':6,'CAL':7,'MAX':8}
pop['site_num'] = pop["site"].map(SITE_MAP)


assert sum(pop.dx_num == 0) == 197
assert sum(pop.dx_num == 1) == 170


# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
