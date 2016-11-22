# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:17:39 2016

@author: ad247405
"""

import os
import numpy as np
import pandas as pd

#import proj_classif_config
GROUP_MAP = {'N': 0, 'Y': 1}
GENDER_MAP = {'F': 0, 'H': 1}
COHORT_MAP = {'eugei': 0, 'icaar': 1}



BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_icaar-eugei/documents/icaar_eugei_images_correspondanceV2 CMLF.xls'
INPUT_SUBJECTS_LIST_FILENAME = '/neurospin/brainomics/2016_icaar-eugei/list_subjects/list_VBM_map.txt'
OUTPUT_CSV_ICAAR = os.path.join(BASE_PATH,"results","VBM","ICAAR","population.csv")
OUTPUT_CSV_ICAAR_EUGEI = os.path.join(BASE_PATH,"results","VBM","ICAAR+EUGEI","population.csv")


#ICAAR + EUGEI population file
##############################################################################
# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (76, 14)
# Read subjects with image
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
                               
clinic["path_VBM"] = input_subjects

pop = clinic[:][["group_inclusion", "group_outcome", "sex", "age","path_VBM","cohort"]]
assert  pop.shape == (76,6)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)
pop['cohort.num'] = pop["cohort"].map(COHORT_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR_EUGEI, encoding='utf-8' ) 
##############################################################################


#ICAAR only population file
##############################################################################
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (76, 14)
# Read subjects with image
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
                               
clinic["path_VBM"] = input_subjects
clinic = clinic[clinic.cohort!='eugei']
assert  clinic.shape == (55, 20)
pop = clinic[:][["group_inclusion", "group_outcome", "sex", "age","path_VBM","cohort"]]
assert  pop.shape == (55,5)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)



# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR, encoding='utf-8' ) 
####################################

##########################################