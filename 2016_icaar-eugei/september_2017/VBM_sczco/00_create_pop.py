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

#import proj_classif_config
GROUP_MAP = {'N': 0, 'Y': 1}
GENDER_MAP = {'F': 0, 'H': 1}
COHORT_MAP = {'eugei': 0, 'icaar': 1}



BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei/september_2017'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_icaar-eugei/documents/icaar_eugei_images_correspondanceV2 CMLF.xls'
OUTPUT_CSV_ICAAR = os.path.join(BASE_PATH,"VBM","ICAAR_sczco","population.csv")
INPUT_DATA = "/neurospin/brainomics/2016_icaar-eugei/data/ICAAR-EUGEI/ICAAR"
OUTPUT_CSV_ICAAR_WITH_SCORES = os.path.join(BASE_PATH,"VBM","ICAAR_sczco","population+scores.csv")


#ICAAR only population file
##############################################################################
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (76, 14)

# Read subjects with image
paths = glob.glob(os.path.join(INPUT_DATA,"Acquistion_complete/*/*/mwc1*.nii"))
subjects = list()
for i in range(len(paths)):
    subjects.append(os.path.split(os.path.split(os.path.split(paths[i])[0])[0])[1])

input_subjects_vbm = pd.DataFrame(subjects, columns=["image"])
input_subjects_vbm["path_VBM"] = paths

assert input_subjects_vbm.shape == (55, 2)


pop = clinic.merge(input_subjects_vbm,on = "image")
pop = pop[["image","Code ICAAR","group_inclusion", "group_outcome", "sex", "age","path_VBM","cohort"]]

assert  pop.shape == (55,8)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
assert  pop.shape == (55,10)
# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR, encoding='utf-8' )

INPUT_SCORES = "/neurospin/brainomics/2016_icaar-eugei/documents/ICAAR_2017_09transmis.xls"
scores = pd.read_excel(INPUT_SCORES)
scores["Code ICAAR"] = scores["PATIENT"]


pop_with_scores = pop.merge(scores,on = "Code ICAAR")
assert  pop_with_scores.shape == (53,119)

pop_with_scores.to_csv(OUTPUT_CSV_ICAAR_WITH_SCORES, encoding='utf-8' )


##############################################################################
