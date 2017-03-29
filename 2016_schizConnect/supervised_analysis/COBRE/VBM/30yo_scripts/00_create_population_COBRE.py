#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 16:20:11 2017

@author: ad247405
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import glob

INPUT_DATA = "/neurospin/abide/schizConnect/data_nifti_format/COBRE"
BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/completed_schizconnect_metaData_1829.csv'
OUTPUT_CSV = os.path.join(BASE_PATH,"population_30yo.csv")





# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
clinic = clinic[clinic.study=="COBRE"]


# Read subjects with image
subjects = list()
date = list()
paths = glob.glob(os.path.join(INPUT_DATA,"MRN/*/*/mwc1average_T1.nii"))
for i in range(len(paths)):
    subjects.append(os.path.split(os.path.split(os.path.split(paths[i])[0])[0])[1])
    date.append(os.path.split(os.path.split(paths[i])[0])[1])

    
input_subjects_vbm = pd.DataFrame(subjects, columns=["subjectid"])
input_subjects_vbm["path_VBM"] = paths
input_subjects_vbm["date"] = date
assert input_subjects_vbm.shape == (167, 3)                       



input_subjects_vbm["age"] = "NaN"
input_subjects_vbm["sex"] = "NaN"
input_subjects_vbm["dx"] = "NaN"
input_subjects_vbm["site"] = "NaN"
for s in input_subjects_vbm.subjectid:
     age =  clinic.loc[clinic.subjectid == s,"age"].values[0]   
     input_subjects_vbm.loc[input_subjects_vbm.subjectid == s,"age"] = int(age)
     
     sex = clinic.loc[clinic.subjectid == s,"sex"].values[0]   
     input_subjects_vbm.loc[input_subjects_vbm.subjectid == s,"sex"] = sex
     
     dx =clinic.loc[clinic.subjectid == s,"dx"].values[0]   
     input_subjects_vbm.loc[input_subjects_vbm.subjectid == s,"dx"] = dx

     site = clinic.loc[clinic.subjectid == s,"imaging_protocol_site"].values[0]
     input_subjects_vbm.loc[input_subjects_vbm.subjectid == s,"site"] = site

assert  input_subjects_vbm.shape == (167, 7)
pop = input_subjects_vbm
assert  pop.shape == (167, 7)


# Map group
DX_MAP = {'No_Known_Disorder': 0, 'Sibling_of_No_Known_Disorder':0 ,'Schizophrenia_Strict': 1}
pop['dx_num'] = pop["dx"].map(DX_MAP)

SEX_MAP = {'male': 0, 'female': 1}
pop['sex_num'] = pop["sex"].map(SEX_MAP)



#len(pop.subjectid.unique()) only 270 different subjects
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
  
pop = pop[pop.age<30]


assert sum(pop.dx_num.values==0) == 26
assert sum(pop.dx_num.values==1) == 31
assert sum(pop.sex_num.values==0) == 46
assert sum(pop.sex_num.values==1) == 11





# Save population information
pop.to_csv(OUTPUT_CSV, index=False)

##############################################################################
