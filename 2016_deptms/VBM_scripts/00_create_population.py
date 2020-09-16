# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:35:20 2016

@author: ad247405
"""

import os
import numpy as np
import pandas as pd

GROUP_MAP = {'N': 0, 'Y': 1}


BASE_PATH = '/neurospin/brainomics/2016_deptms'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_deptms/deptms_info.csv'
INPUT_SUBJECTS_LIST_FILENAME = '/neurospin/brainomics/2016_deptms/analysis/VBM/preproc_VBM/list_subjects/list_VBM_map.txt'
OUTPUT_CSV = os.path.join(BASE_PATH,"analysis","VBM","population.csv")




##############################################################################
# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

assert  clinic.shape == (34,32)



# Read subjects with image
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST_FILENAME, sep=" ",
                               header=None)
                               
clinic["path_VBM"] = input_subjects

pop = clinic[:][['subject','Response','Age', 'Sex',"path_VBM"]]
assert  pop.shape == (34, 5)


# Map group
print([[lab, np.sum(pop["Response"] == lab)] for lab in np.unique(pop["Response"])])
# [['N', 16], ['Y', 18]]
pop['Response.num'] = pop["Response"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)

##############################################################################
