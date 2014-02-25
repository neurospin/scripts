# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:06:57 2014

@author: md238665

Create files for one-sample t-test with SPM.
This analysis will create the mask.

We don't need QC because the template folder contains the list of selected
subjects.

"""


import os
import glob

import numpy as np
import pandas as pd

BASE_PATH = "/neurospin/brainomics/2013_adni"

CLINIC_PATH = os.path.join(BASE_PATH, "clinic")
INPUT_CLINIC_FILE = os.path.join(CLINIC_PATH, "adni510_m18_nonnull_groups.csv")

INPUT_TEMPLATE_PATH = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC_MCIc-AD")
INPUT_SUBJECTS_LIST = os.path.join(INPUT_TEMPLATE_PATH,
                                   "subject_list.txt")
INPUT_IMAGE_PATH = os.path.join(INPUT_TEMPLATE_PATH,
                                "registered_images")
INPUT_IMAGEFILE_FORMAT = os.path.join(INPUT_IMAGE_PATH,
                                      "smw{PTID}*_Nat_dartel_greyProba.nii")

OUTPUT_PATH = os.path.join(BASE_PATH,
                           "proj_predict_MMSE",
                           "SPM",
                           "template_FinalQC_MCIc-AD")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "spm_file.txt")

# Read clinic data
m18_clinic = pd.read_csv(INPUT_CLINIC_FILE,
                         index_col=0)

# Read input subjects
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST, sep=" ",
                               header=None)
input_subjects = input_subjects[1].map(lambda x: x[0:10])

output_file = open(OUTPUT_FILE, "w")
print >> output_file, "[Scans]"
for PTID in input_subjects:
    #print "Subject", PTID
    imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=PTID)
    #print imagefile_pattern
    imagefile_name = glob.glob(imagefile_pattern)[0]
    print >> output_file, imagefile_name

print >> output_file
print >> output_file, "[MMSE]"
mmse_str = " ".join([str(m18_clinic['MMSE'].loc[PTID]) for PTID in input_subjects])
print >> output_file, mmse_str
#for PTID in input_subjects:
#    print >> output_file, m18_clinic['MMSE'].loc[PTID]

output_file.close()
