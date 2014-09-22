# -*- coding: utf-8 -*-

"""
Created on Thu july  22 10:46:01 2014

@author: christophe

Extract some of clinical information (SEX, SCANNER and AGEATMRI) and maps
the categorical variables.
"""

## Build population for the project 2014_bd_dwi

import os
import pandas as pd

from collections import OrderedDict

from brainomics import pandas_utilities as pu


#loading files
BASE_PATH =  "/volatile/share/2014_bd_dwi"

#BASE_PATH =  "/volatile/share/2014_bd_dwi/all_FA/nii/stats"


INPUT_IMAGE_INDEX = os.path.join(BASE_PATH, "clinic", "image_list.txt") # file list of images in order from josselin
INPUT_CSV_BASE = os.path.join(BASE_PATH, "clinic", "BD_clinic.csv") # clinic file from josselin
OUTPUT_CSV = os.path.join(BASE_PATH, "population.csv")

#OUTPUT = os.path.join(BASE_PATH, "DATA")

#############################################################################
## Build population file

# Read image order and convert to match the ID in clinic file
image_id = list()
for l in open(INPUT_IMAGE_INDEX).readlines():
    l = l.replace("\n", "")
    l = l.replace("P_S_", "")
    if l.find("C_") == 0:
        l = l[:-4]
    if l.find("M_H_C6HB") == 0:  # M_H_C6HB02 => HB_002
        l = l.replace("M_H_C6HB", "HB_0")
    if l.find("M_H_C6HK") == 0:  # M_H_C6HK01 => HK_001
        l = l.replace("M_H_C6HK", "HK_0")
    if l.find("M_Z_") == 0:  # M_Z_103843 => MZ103843
        l = l.replace("M_Z_", "MZ")
    image_id.append(l)

# Create a dataframe of image position
images = pd.DataFrame(range(len(image_id)), columns=['SLICE'],
                      index=image_id)

# Read some columns clinic data
clinic = pd.read_csv(INPUT_CSV_BASE,
                     index_col=0)
clinic = clinic[["BD_HC", "SCANNER", "AGEATMRI", "SEX"]]

# Map the sex from (1, 2) where 1 is for male and 2 is for female
# to (-1, 1) where 1 is for the female, -1 is for the male
mappings_NC = {'SEX': OrderedDict([(1, -1),
                                   (2, 1)])}
clinic = pu.numerical_coding(clinic, mappings_NC)

# Dummy Coding of scanners
mappings_DC = {'SCANNER': OrderedDict([(1, 1),
                                       (2, 2),
                                       (3, 3)])}

clinic = pu.indicator_variables(clinic, mappings_DC)

# Merge dataframes
population = pd.merge(clinic, images,
                      right_index = True,
                      left_index = True)
assert(population.shape[0] == 194)

# Print ID of subjects for which clinic or image is missing
in_clinic_not_in_image = clinic.index[~clinic.index.isin(images.index)]
for ID in in_clinic_not_in_image:
    print "ID", ID, "matched 0 images"
assert(len(in_clinic_not_in_image)==10)
#ID 204776 matched 0 images
#ID 209861 matched 0 images
#ID 211297 matched 0 images
#ID 212485 matched 0 images
#ID 213141 matched 0 images
#ID 23012 matched 0 images
#ID 27308 matched 0 images
#ID 31108 matched 0 images
#ID C_cn00018 matched 0 images
#ID MZ150743 matched 0 images

in_image_not_in_clinic = images.index[~images.index.isin(clinic.index)]
for ID in in_image_not_in_clinic:
    print "ID", ID, "matched 0 clinic"
assert(len(in_image_not_in_clinic)==6)
#ID 023012 matched 0 clinic
#ID C_pb090397 matched 0 clinic
#ID 031108 matched 0 clinic
#ID C_gb100352 matched 0 clinic
#ID 027308 matched 0 clinic
#ID 214469 matched 0 clinic

population.to_csv(OUTPUT_CSV)

#