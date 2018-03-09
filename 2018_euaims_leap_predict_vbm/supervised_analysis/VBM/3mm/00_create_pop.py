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


BASE_PATH = '/neurospin/brainomics/2018_euaims_leap_predict_vbm'
INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH,"documents","Demographics_LEAP_V01_2018-02-16_17_07_03.csv")
OUTPUT_CSV = os.path.join(BASE_PATH,"results","VBM","3mm","population.csv")
INPUT_DATA = "/neurospin/brainomics/2018_euaims_leap_predict_vbm/data/processed/LEAP_V01"


# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME,delimiter = ";",header=1,dtype={'ID': object})
number_subjects = clinic.shape[0]
assert clinic.shape ==  (748, 63)


# Read subjects with image
subjects = list()
path_subjects= list()
paths = glob.glob(os.path.join(INPUT_DATA,"*/anatomy/mwc1*_3mm.nii"))
for i in range(len(paths)):
    path_subjects.append(paths[i])
    subjects.append(paths[i][75:87])



pop = pd.DataFrame(subjects, columns=["ID"])
pop["path_VBM"] = path_subjects
assert pop.shape == (766, 2)


list1 = pop["ID"].values
list2 = clinic["ID"].values
print(list(set(list1) - set(list2)))


pop = pop.merge(clinic, on = "ID")
assert pop.shape == (649, 64)

pop = pop[pop["group"] != 3.0]
pop = pop[pop["group"] != 4.0]
assert pop.shape == (577, 64)

# Map group
#See Demographic_LEAP.pdf: Group 1 are Controls, Group 2 are ASD
DX_MAP = {1:0,2:1,3:2,4:3}
SEX_MAP = {-1:0,1:1}
#0:female, 1: male

pop['dx_num'] = pop["group"].map(DX_MAP)
pop['sex_num'] = pop["sex"].map(SEX_MAP)

pop.replace(to_replace = '999',value="nan",inplace=True)
pop.replace(to_replace = '777',value="nan",inplace=True)

age = pd.to_numeric(pop["age"],errors='coerce').values
pop.loc[pop["age"] == "nan","age"] = np.nanmean(age)


# Save population information
pop.to_csv(OUTPUT_CSV , index=False)

##############################################################################
