# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:11:07 2016

@author: ad247405
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:42:15 2016

@author: ad247405
"""

import os
import numpy as np
import pandas as pd

#import proj_classif_config
GROUP_MAP = {1: 1, '2a': 2, '2b': 3, 3 : 0}
GENDER_MAP = {'f': 0, 'm': 1}




BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_AUSZ/clinic.xlsx'
INPUT_SUBJECTS_LIST_FILENAME = '/neurospin/brainomics/2016_AUSZ/list_VBM_WM.txt'
OUTPUT_CSV = os.path.join(BASE_PATH,"results","VBM","WM","population.csv")


#ICAAR + EUGEI population file
##############################################################################
# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
assert  clinic.shape == (119, 11)



#Drop rows corresponding to excluded subjects
clinic = clinic.drop(clinic[clinic['ID']=='GS130218'].index)
clinic = clinic.drop(clinic[clinic['ID']=='BL130250'].index)
clinic = clinic.drop(clinic[clinic['ID']=='SR160570'].index)
clinic = clinic.drop(clinic[clinic['ID']=='SR160602'].index)
assert  clinic.shape == (115, 11)


# Read subjects with image
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
                               
clinic["path_VBM"] = input_subjects

pop = clinic[:][["groupe", "Sex", "Age","path_VBM"]]
assert  pop.shape == (115,4)

# Map group
pop['group.num'] = pop["groupe"].map(GROUP_MAP)
pop['sex.num'] = pop["Sex"].map(GENDER_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV, encoding='utf-8' ) 
##############################################################################
