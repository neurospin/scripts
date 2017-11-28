
"""

Creates a CSV file for the population.

"""
import os
import numpy as np
import pandas as pd
import re
import glob


BASE_PATH = "/neurospin/brainomics/2017_memento/analysis/FS"
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2017_memento/documents/MEMENTO_BASELINE_ML_for_catidb.csv'
INPUT_FS = "/neurospin/brainomics/2017_memento/analysis/FS/freesurfer_assembled_data_fsaverage"

OUTPUT_CSV = "/neurospin/brainomics/2017_memento/analysis/FS/population.csv"


# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-7]
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = os.path.basename(path)[:-7]
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["usubjid", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (2166, 3)


# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="usubjid")
assert  clinic.shape == (2164, 27)

# Save population information
clinic.to_csv(OUTPUT_CSV, index=False)