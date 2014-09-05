# -*- coding: utf-8 -*-
"""

@author: edouard.duchesnay@cea.fr

Creates a CSV file for the population.
=> intersection of subject_list.txt and adnimerge_simplified.csv of in CTL or AD

"""
import os
import numpy as np
import pandas as pd
import re
import glob

#import proj_classif_config
GROUP_MAP = {'CTL': 0, 'MCIc': 1}

BASE_PATH = "/neurospin/brainomics/2013_adni"
INPUT_CLINIC_FILENAME = os.path.join(BASE_PATH, "clinic", "adnimerge_simplified.csv")
INPUT_FS = "/neurospin/brainomics/2013_adni/freesurfer_assembled_data"

INPUT_VBM = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC",
                                   "subject_list.txt")

OUTPUT_CSV = os.path.join(BASE_PATH,
                          "MCIc-CTL-FS",
                          "population.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
        os.makedirs(os.path.dirname(OUTPUT_CSV))

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

# Read free surfer assembled_data
input_subjects_fs = dict()
paths = glob.glob(INPUT_FS+"/*_lh.mgh")
for path in paths:
    subject = re.search("^([0-9]{3}_S_[0-9]{4}).+_lh", os.path.basename(path)).group(1)
    input_subjects_fs[subject] = [path]

paths = glob.glob(INPUT_FS+"/*_rh.mgh")
for path in paths:
    subject = re.search("^([0-9]{3}_S_[0-9]{4}).+_rh", os.path.basename(path)).group(1)
    input_subjects_fs[subject].append(path)

# Remove if some hemisphere is missing
input_subjects_fs = [[k]+input_subjects_fs[k] for k in input_subjects_fs if len(input_subjects_fs[k]) == 2]

input_subjects_fs = pd.DataFrame(input_subjects_fs,  columns=["PTID", "mri_path_lh",  "mri_path_rh"])
assert input_subjects_fs.shape == (509, 3)

# Read subjects with image with VBM and keep only fs subejct wirh vbm
input_subjects_vbm = pd.read_table(INPUT_VBM, sep=" ", header=None)
input_subjects_vbm = [x[:10] for x in input_subjects_vbm[1]]
input_subjects_fs = input_subjects_fs[input_subjects_fs["PTID"].isin(input_subjects_vbm)]
assert input_subjects_fs.shape == (455, 3)

# intersect with subject with image
clinic = clinic.merge(input_subjects_fs, on="PTID")
assert  clinic.shape == (455, 96)

# Extract sub-population 
# MCIc = MCI at bl converion to AD within 800 days
TIME_TO_CONV = 800#365 * 2
mcic_m = (clinic["DX.bl"] == "MCI") &\
       (clinic.CONV_TO_AD < TIME_TO_CONV) &\
       (clinic["DX.last"] == "AD")
assert np.sum(mcic_m) == 81

mcic = clinic[mcic_m][['PTID', 'AGE', 'PTGENDER', "DX.bl", "mri_path_lh",  "mri_path_rh"]]
mcic["DX"] = "MCIc"

# CTL: CTL at bl no converion to AD
ctl_m = (clinic["DX.bl"] == "CTL") &\
      (clinic["DX.last"] == "CTL")
assert np.sum(ctl_m) == 120

ctl = clinic[ctl_m][['PTID', 'AGE', 'PTGENDER', "DX.bl", "mri_path_lh",  "mri_path_rh"]]
ctl["DX"] = "CTL"

pop = pd.concat([mcic, ctl])
assert pop.shape == (201, 7)

# Map group
pop['DX.num'] = pop["DX"].map(GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
