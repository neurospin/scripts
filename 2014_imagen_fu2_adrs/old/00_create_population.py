# -*- coding: utf-8 -*-
"""

@author: edouard.duchesnay@cea.fr



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


OUTPUT_CSV = os.path.join(BASE_PATH,
                          "ADRS2",
                          "population.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

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
clinic["adrs"] = clinic.adrs1 + clinic.adrs2 + clinic.adrs3 + clinic.adrs4 +\
    clinic.adrs5 + clinic.adrs5 + clinic.adrs7 + clinic.adrs8 + clinic.adrs9 +\
    clinic.adrs10

pop = clinic[["Subject", "Gender",  "ImagingCentreID", "Age for timestamp", "adrs", "mri_path"]]
pop.columns = ["Subject", "Gender",  "ImagingCentreID", "Age", "adrs", "mri_path"]

assert pop.shape == (1082, 6)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
