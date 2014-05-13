# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:06:57 2014

@author: md238665

Create files for two-sample t-test with SPM (CTL vs AD).
This analysis will create the mask.

We don't need QC because the template folder contains the list of selected
subjects.

We also create a CSV file for the population.

"""


import os
import glob

import numpy as np
import pandas as pd

import proj_classif_config

BASE_PATH = proj_classif_config.BASE_PATH

CLINIC_PATH = os.path.join(BASE_PATH, "clinic")
INPUT_CLINIC_FILE = os.path.join(CLINIC_PATH, "adni510_bl_groups.csv")
INPUT_GROUPS = ['control', 'AD']

INPUT_TEMPLATE_PATH = os.path.join(BASE_PATH,
                                   "templates",
                                   "template_FinalQC")
INPUT_SUBJECTS_LIST = os.path.join(INPUT_TEMPLATE_PATH,
                                   "subject_list.txt")
INPUT_IMAGE_PATH = os.path.join(INPUT_TEMPLATE_PATH,
                                "registered_images")
INPUT_IMAGEFILE_FORMAT = os.path.join(INPUT_IMAGE_PATH,
                                      "mw{PTID}*_Nat_dartel_greyProba.nii")

OUTPUT_BASE_PATH = proj_classif_config.PROJ_PATH
OUTPUT_CSV = os.path.join(OUTPUT_BASE_PATH,
                          "population.csv")
OUTPUT_PATH = os.path.join(OUTPUT_BASE_PATH,
                           "SPM",
                           "template_FinalQC_CTL_AD")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "spm_file.txt")

# Read clinic data
clinic = pd.read_csv(INPUT_CLINIC_FILE,
                     index_col=0)

# Read input subjects
input_subjects = pd.read_table(INPUT_SUBJECTS_LIST, sep=" ",
                               header=None)
input_subjects = input_subjects[1].map(lambda x: x[0:10])

# Extract sub-population
is_input_groups = clinic['Group.ADNI'].isin(INPUT_GROUPS)
is_in_input = clinic.index.isin(input_subjects)
pop = clinic[is_input_groups & is_in_input]
n = len(pop)
print "Found", n, "subjects in", INPUT_GROUPS

# Map group
pop['Group.num'] = pop['Group.ADNI'].map(proj_classif_config.GROUP_MAP)

# Save population information
pop.to_csv(OUTPUT_CSV)

# Create file
output_file = open(OUTPUT_FILE, "w")
for group in INPUT_GROUPS:
    group_num = proj_classif_config.GROUP_MAP[group]
    print >> output_file, "[{group_num} ({group})]".format(group_num=group_num,
                                                           group=group)
    is_sub_pop = pop['Group.num'] == group_num
    sub_pop = pop[is_sub_pop]
    n_sub = len(sub_pop)
    print "Found", n_sub, "in", group
    for ptid in sub_pop.index:
        print ptid
        imagefile_pattern = INPUT_IMAGEFILE_FORMAT.format(PTID=ptid)
        imagefile_name = glob.glob(imagefile_pattern)[0]
        print >> output_file, imagefile_name
    print >> output_file

output_file.close()

# Display some statistics
import itertools
levels = list(itertools.product(INPUT_GROUPS, ['training', 'testing']))
index = pd.MultiIndex.from_tuples(levels,
                                  names=['Class', 'Set'])
count = pd.DataFrame(columns=['Count'], index=index)
for i, (group, sample) in enumerate(levels):
    is_in = (pop['Group.ADNI'] == group) & (pop['Sample'] == sample)
    count['Count'].loc[group, sample] = is_in.nonzero()[0].shape[0]
