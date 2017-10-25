#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:04:05 2017

@author: ad247405
"""



import os
import numpy as np
import pandas as pd
import glob

INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/data/VBM/orig"
BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/results/VBM'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/analysis/PRAGUE/document/prague_demographics2.xls'
OUTPUT_CSV = os.path.join(BASE_PATH,"population.csv")




# Read clinic data

# Read clinic data
clinic_patients = pd.read_excel(INPUT_CLINIC_FILENAME,sheetname = 0)
clinic_controls = pd.read_excel(INPUT_CLINIC_FILENAME,sheetname = 1,skip_footer = 51)
number_patients = clinic_patients.shape[0]
number_controls = clinic_controls.shape[0]

clinic = pd.DataFrame()
clinic["code"] = list(clinic_patients['PATIENT\ncode'].values) + list(clinic_controls['CONTROL\ncode'].values)
clinic["age"] = list(clinic_patients['Age\n(  MRI scan)  '].values) + list(clinic_controls['Age\n(  MRI scan)  )  '].values)
clinic["sex"] = list(clinic_patients['Sex\n1-man\n2-woman'].values) + list(clinic_controls['Sex\n1-man\n2-woman'].values)
clinic["laterality"] = list(clinic_patients['Laterality (EHI)'].values) + list(clinic_controls['Laterality (EHI)'].values)
clinic["DX"] = np.nan
clinic["DX"][0:number_patients] = list(clinic_patients['Dg.'].values)
clinic["DX"][number_patients:] = 0.0
clinic["onset_first_symptoms"] = np.nan
clinic["onset_first_symptoms"][0:number_patients] = list(clinic_patients['onset of first symptoms'].values)
clinic["illness_duration"] = np.nan
clinic["illness_duration"][0:number_patients] = list(clinic_patients['ilness duration in months)'].values)
clinic["DUP"] = np.nan
clinic["DUP"][0:number_patients] = list(clinic_patients['DUP '].values)
assert clinic.shape == (190, 8)


# Read subjects with image
subjects = list()
path_subjects= list()
paths = glob.glob(os.path.join(INPUT_DATA,"mwc1*.nii"))
for i in range(len(paths)):
    path_subjects.append(paths[i])
    subjects.append("ESO"+os.path.basename(paths[i])[-10:-4]) #AJOUTER P OR C to match thec ode



pop = pd.DataFrame(subjects, columns=["code"])
pop["path_VBM"] = path_subjects
assert pop.shape == (133, 2)


pop = pop.merge(clinic, on = "code")
assert pop.shape == (133, 9)


# Map group
# Map group
DX_MAP = {"F23.0":1,"F23.1":1,"F20.0":1,"F23.1, F15.5":1,0.0:0}
SEX_MAP = {1:0,2:1}
#0:male, 1: female

pop['dx_num'] = pop["DX"].map(DX_MAP)
pop['sex_01code'] = pop["sex"].map(SEX_MAP)


# Save population information
pop.to_csv(OUTPUT_CSV , index=False)

##############################################################################
