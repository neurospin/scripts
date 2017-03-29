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


BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP"
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/analysis/VIP/sujets_series.xls'
INPUT_FS = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/freesurfer_assembled_data_fsaverage"

OUTPUT_CSV = os.path.join(BASE_PATH,"Freesurfer","population.csv")


# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
clinic = clinic[clinic.diagnostic !=2]
clinic = clinic[clinic.diagnostic !=4]
clinic = clinic[clinic.diagnostic.isnull()!= True]


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

                     
input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["nip", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (80, 3)


# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="nip")
assert  clinic.shape == (80, 25)



pop = clinic[['nip','diagnostic','sexe', "mri_path_lh",  "mri_path_rh","ddn","acq date"]]
assert  pop.shape == (80, 7)

from datetime import datetime, timedelta

pd.to_datetime(pop["ddn"])
pd.to_datetime(pop["acq date"],format='%Y%m%d')
pop.age_days = pd.to_datetime(pop["acq date"],format='%Y%m%d') - pd.to_datetime(pop["ddn"])
pop['age'] = pop.age_days/ timedelta(days=365)



# Map group
# Map group
DX_MAP = {1.0: 0, 3.0: 1}
SEX_MAP = {1.0: 0, 2.0: 1.0}
pop['dx_num'] = pop["diagnostic"].map(DX_MAP)
pop['sex_num'] = pop["sexe"].map(SEX_MAP)
pop['site'] = "vip"

pop = pop[["mri_path_lh","mri_path_rh","age","sex_num","dx_num","site"]]
# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
