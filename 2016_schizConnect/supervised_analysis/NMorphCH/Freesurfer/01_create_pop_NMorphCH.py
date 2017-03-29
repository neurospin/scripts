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


BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH"
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/completed_schizconnect_metaData_1829.csv'
INPUT_FS = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/Freesurfer/data/freesurfer_assembled_data_fsaverage"

OUTPUT_CSV = os.path.join(BASE_PATH,"Freesurfer","population.csv")


# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
clinic = clinic[clinic.study=="NMorphCH"]
# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = os.path.basename(path)[20:-7]
    date = os.path.basename(path)[9:-(len(subject)+8)]
    id = os.path.basename(path)[9:-7]
    input_subjects_fs[id] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = os.path.basename(path)[20:-7]
    date = os.path.basename(path)[9:-(len(subject)+8)]
    id = os.path.basename(path)[9:-7]
    input_subjects_fs[id].append(path)
    input_subjects_fs[id].append(subject)
    input_subjects_fs[id].append(date)

    
# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 4]
input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["id", "mri_path_lh",  "mri_path_rh","subjectid","date"])
assert input_subjects_fs.shape == (110, 5)

input_subjects_fs["age"] = "NaN"
input_subjects_fs["sex"] = "NaN"
input_subjects_fs["dx"] = "NaN"
input_subjects_fs["site"] = "NaN"
for s in input_subjects_fs.id:
     subject = input_subjects_fs[input_subjects_fs.id == s].subjectid.item()
     age =  clinic.loc[clinic.subjectid == subject,"age"].values[0]   
     input_subjects_fs.loc[input_subjects_fs.id == s,"age"] = int(age)
     
     sex = clinic.loc[clinic.subjectid == subject,"sex"].values[0]   
     input_subjects_fs.loc[input_subjects_fs.id == s,"sex"] = sex
     
     dx =clinic.loc[clinic.subjectid == subject,"dx"].values[0]   
     input_subjects_fs.loc[input_subjects_fs.id == s,"dx"] = dx

     site = clinic.loc[clinic.subjectid == subject,"imaging_protocol_site"].values[0]
     input_subjects_fs.loc[input_subjects_fs.id == s,"site"] = site
    


assert  input_subjects_fs.shape == (110, 9)


pop = input_subjects_fs
assert  pop.shape == (110, 9)


# Map group
DX_MAP = {'No_Known_Disorder': 0, 'Schizophrenia_Strict': 1}
pop['dx_num'] = pop["dx"].map(DX_MAP)

SEX_MAP = {'male': 0, 'female': 1}
pop['sex_num'] = pop["sex"].map(SEX_MAP)

#pop.site.unique()  only one site. No need for this variable




#len(pop.subjectid.unique()) only 269 different subjects
#retain only last scan for each subject (most recent)
to_delete_vol_index = []
for s in pop.subjectid.unique() :
    volumes = pop[pop.subjectid == s]
    if len(volumes) > 1:
        volumes['pd_date'] = pd.to_datetime(volumes["date"])
        index = volumes[volumes['pd_date'] != volumes['pd_date'].max() ].index.tolist()
        for i in range(len(index)):
            to_delete_vol_index.append(index[i])

          
pop.drop(pop.index[[to_delete_vol_index]],inplace=True)

assert pop.shape[0] == len(pop.subjectid.unique())  
  

assert sum(pop.dx_num.values==0) == 37 #controls
assert sum(pop.dx_num.values==1) == 36 #SCZ
assert sum(pop.sex_num.values==0) == 39 #Male
assert sum(pop.sex_num.values==1) == 34 #female


# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
