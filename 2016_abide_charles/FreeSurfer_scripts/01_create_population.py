# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:46:52 2016

@author: ad247405
"""

"""

Creates a CSV file for the population.


"""
import os
import numpy as np
import pandas as pd
import re
import glob

#import proj_classif_config
GROUP_MAP = {'N': 0, 'Y': 1}

BASE_PATH = '/neurospin/brainomics/2016_deptms'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_deptms/deptms_info.csv'
INPUT_FS = "/neurospin/brainomics/2016_deptms/results/Freesurfer/freesurfer_assembled_data_fsaverage"

OUTPUT_CSV = os.path.join(BASE_PATH,"results","Freesurfer","population.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = re.search("^([A-Z]{1}[0-9]{4,5})+_lh", os.path.basename(path)).group(1)
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = re.search("^([A-Z]{1}[0-9]{4,5})+_rh", os.path.basename(path)).group(1)
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["subject", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (34, 3)


# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="subject")
assert  clinic.shape == (34, 34)



pop = clinic[['subject','Response','Age', 'Sex', "mri_path_lh",  "mri_path_rh"]]
assert  pop.shape == (34, 6)


# Map group
pop['Response.num'] = pop["Response"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
