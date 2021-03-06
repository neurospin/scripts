# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 17:27:51 2014

@author: cp243490
"""

import os
import numpy as np
import pandas as pd
import glob
import re

#import proj_classif_config
#GROUP_MAP = {'CTL': 0, 'AD': 1}

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs"

INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "imagen_adrs_fu1_20140528.csv")
INPUT_SUBJECTS_LIST_FILENAME = "/neurospin/brainomics/2012_imagen_shfj/VBM/new_segment_spm8/smwc1*"

OUTPUT_POP_CSV = os.path.join(BASE_PATH,            "ADRS_population",
                                              "population.csv")
OUTPUT_POP_DISCRETE_CSV = os.path.join(BASE_PATH,   "ADRS_population",
                                              "population_discrete-adrs.csv")


if not os.path.exists(os.path.dirname(OUTPUT_POP_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_POP_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
#Psytool QC
#The Psytools Valid flag is shown as ‘invalid’ if at least one of the following cases apply:
#Automated flags for context questions:
#ts_2 = "1", ts_4 = "3", ts_5 ="4" or ts_5 = "5"
assert np.sum(clinic.ts_2 == 1) == 80
assert np.sum(clinic.ts_4 == 3) == 8
#np.sum(clinic.ts_5 == 4)  MISSING
#np.sum(clinic.ts_5 == 5)
clinic = clinic[(clinic.ts_2 != 1) | (clinic.ts_4 != 3)]
assert  clinic.shape == (1216, 24)
# Read subjects with image
input_subjects_filenames = glob.glob(INPUT_SUBJECTS_LIST_FILENAME)

# extract subject id from filename
regexp = re.compile('smwc1([0-9]+)s.+.nii')
input_subjects = [int(regexp.findall(os.path.basename(x))[0]) for x in input_subjects_filenames]

subjects_images = pd.DataFrame(dict(Subject=input_subjects, mri_path=input_subjects_filenames))

# intersect with clinic with images
clinic = pd.merge(clinic, subjects_images)

assert clinic.shape == (1082, 25)

# Verifier items positi varie dans meme sens
clinic.adrs1[clinic.adrs1 == 2] = 0
clinic.adrs2[clinic.adrs2 == 2] = 0
clinic.adrs3[clinic.adrs3 == 2] = 0
clinic.adrs4[clinic.adrs4 == 2] = 0
clinic.adrs5[clinic.adrs5 == 2] = 0
clinic.adrs6[clinic.adrs6 == 2] = 0
clinic.adrs7[clinic.adrs7 == 2] = 0
clinic.adrs8[clinic.adrs8 == 2] = 0
clinic.adrs9[clinic.adrs9 == 2] = 0
clinic.adrs10[clinic.adrs10 == 2] = 0

clinic["adrs"] = clinic.adrs1 + clinic.adrs2 + clinic.adrs3 + clinic.adrs4 +\
    clinic.adrs5 + clinic.adrs6 + clinic.adrs7 + clinic.adrs8 + clinic.adrs9 +\
    clinic.adrs10

pop = clinic[["Subject", "Gender",  "ImagingCentreID",
              "Age for timestamp", "adrs", "mri_path"]]
pop.columns = ["Subject", "Gender",  "ImagingCentreID",
               "Age", "adrs", "mri_path"]

# discrete approach
# Comparison of subjects with adrs = 0 and subjects with adrs >= 6
# if adrs = 0 -> adrs_discrete = 0
# if adrs >= 6 -> adrs_discrete = 1
pop_discrete = pop[np.logical_or(pop.adrs == 0, pop.adrs >= 6)]
pop_discrete.adrs[pop.adrs >= 6] = 1

assert pop.shape == (1082, 6)
assert pop_discrete.shape == (618, 6)

# Save population information
pop.to_csv(OUTPUT_POP_CSV, index=False)
pop_discrete.to_csv(OUTPUT_POP_DISCRETE_CSV, index=False)