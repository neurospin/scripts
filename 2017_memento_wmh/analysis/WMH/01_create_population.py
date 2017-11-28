# -*- coding: utf-8 -*-
"""

Creates a CSV file for the population.

"""
import os
import numpy as np
import pandas as pd
import re
import glob


BASE_PATH = "/home/ad247405/git/scripts/2017_memento_wmh/analysis"
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2017_memento/documents/MEMENTO_BASELINE_ML_for_catidb.csv'
INPUT_WMH = "/neurospin/cati/MEMENTO/WMH_registration_MNI_space/3DT1"

OUTPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/population.csv"

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read free surfer assembled_data
input_subjects_wmh = dict()
paths = glob.glob(INPUT_WMH+"/*.nii.gz")
for path in paths:
    subject = os.path.basename(path)[-18:-7]
    input_subjects_wmh[subject] = [path]

# Remove if some  is missing
input_subjects_wmh = [[k]+input_subjects_wmh[k] for k in input_subjects_wmh if len(input_subjects_wmh[k]) == 1]

input_subjects_wmh = pd.DataFrame(input_subjects_wmh,  columns=["usubjid", "wmh_path_lh"])
assert input_subjects_wmh.shape == (2177, 2)


# intersect with subject with image
clinic = clinic.merge(input_subjects_wmh, on="usubjid")
assert  clinic.shape == (2175, 26)


pop = clinic

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)