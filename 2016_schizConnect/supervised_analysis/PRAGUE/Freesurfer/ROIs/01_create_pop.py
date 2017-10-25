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


BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE"
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/document/prague_demographics2.xls'
INPUT_FS = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/data/FS/freesurfer_assembled_data_fsaverage"

OUTPUT_CSV = os.path.join(BASE_PATH,"results","Freesurfer","population.csv")


# Read clinic data
pop_patients = pd.read_excel(INPUT_CLINIC_FILENAME,sheetname = 0)
pop_controls = pd.read_excel(INPUT_CLINIC_FILENAME,sheetname = 1,skip_footer = 51)
number_patients = pop_patients.shape[0]
number_controls = pop_controls.shape[0]

pop = pd.DataFrame()
pop["code"] = list(pop_patients['PATIENT\ncode'].values) + list(pop_controls['CONTROL\ncode'].values)
pop["age"] = list(pop_patients['Age\n(  MRI scan)  '].values) + list(pop_controls['Age\n(  MRI scan)  )  '].values)
pop["sex"] = list(pop_patients['Sex\n1-man\n2-woman'].values) + list(pop_controls['Sex\n1-man\n2-woman'].values)
pop["laterality"] = list(pop_patients['Laterality (EHI)'].values) + list(pop_controls['Laterality (EHI)'].values)
pop["DX"] = np.nan
pop["DX"][0:number_patients] = list(pop_patients['Dg.'].values)
pop["DX"][number_patients:] = 0.0
pop["onset_first_symptoms"] = np.nan
pop["onset_first_symptoms"][0:number_patients] = list(pop_patients['onset of first symptoms'].values)
pop["illness_duration"] = np.nan
pop["illness_duration"][0:number_patients] = list(pop_patients['ilness duration in months)'].values)
pop["DUP"] = np.nan
pop["DUP"][0:number_patients] = list(pop_patients['DUP '].values)


# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = "ESO"+os.path.basename(path)[4:10]
    input_subjects_fs[subject] = [path]


paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = "ESO"+os.path.basename(path)[4:10]
    input_subjects_fs[subject].append(path)


# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]


input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["code", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (133, 3)


# intersect with subject with image
clinic = pop.merge(input_subjects_fs, on="code")
assert  clinic.shape == (132, 10)


# Map group
DX_MAP = {"F23.0":1,"F23.1":1,"F20.0":1,"F23.1, F15.5":1,0.0:0}
clinic['dx_num'] = clinic["DX"].map(DX_MAP)
SEX_MAP = {1:0,2:1}
#0:male, 1: female
clinic['sex_01code'] = clinic["sex"].map(SEX_MAP)

# Save population information
clinic.to_csv(OUTPUT_CSV, index=False)
