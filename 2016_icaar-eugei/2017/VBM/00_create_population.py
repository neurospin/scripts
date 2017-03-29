# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:17:39 2016

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



BASE_PATH = '/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_icaar-eugei/documents/icaar_eugei_images_correspondanceV2 CMLF.xls'
OUTPUT_CSV_ICAAR = os.path.join(BASE_PATH,"VBM","ICAAR","population.csv")
INPUT_DATA = "/neurospin/brainomics/2016_icaar-eugei/data/ICAAR-EUGEI/ICAAR"


#ICAAR only population file
##############################################################################
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (76, 14)
# Read subjects with image

paths = glob.glob(os.path.join(INPUT_DATA,"Acquistion_complete/*/*/site_template_mwc1*.nii"))
subjects = list()
for i in range(len(paths)):
    subjects.append(os.path.split(os.path.split(os.path.split(paths[i])[0])[0])[1])

input_subjects_vbm = pd.DataFrame(subjects, columns=["image"])
input_subjects_vbm["path_VBM"] = paths

assert input_subjects_vbm.shape == (55, 2)                        

    
pop = clinic.merge(input_subjects_vbm,on = "image")   
pop = pop[["group_inclusion", "group_outcome", "sex", "age","path_VBM","cohort"]] 

assert  pop.shape == (55,6)

# Map group
pop['group_outcom.num'] = pop["group_outcome"].map(GROUP_MAP)
pop['sex.num'] = pop["sex"].map(GENDER_MAP)



# Save population information
pop.to_csv(OUTPUT_CSV_ICAAR, encoding='utf-8' ) 
####################################

##########################################