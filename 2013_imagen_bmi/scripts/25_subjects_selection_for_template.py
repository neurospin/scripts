# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 14:18:31 2014

@author: hl237680

This script generates a list of the images to use to construct a template
controlling for normal subjects, that is subjects of normal weight status
and whose neuroanatomic images passed the quality control (grade A).

INPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"

OUTPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/template/
      images_for_normal_template.txt"

676 'normal' subjects among the 1265 for who we have both neuroimaging and
genetic data meet these criteria.
"""


import os
import numpy as np
import pandas as pd
from glob import glob

################################

# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SHFJ_DATA_PATH = os.path.join(CLINIC_DATA_PATH, 'source_SHFJ')
ORIGIN_IMG_DIR = os.path.join(DATA_PATH, 'VBM')
GASER_VBM8_PATH = os.path.join(ORIGIN_IMG_DIR, 'gaser_vbm8')
TEMPLATE_PATH = os.path.join(DATA_PATH, 'template')


################################

# File population.csv containing all clinical data for IMAGEN subjects loaded
# as a dataframe
population_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                 'population.csv'),
                                      sep=',',
                                      index_col=0)

# Select subjects who have a normal weight status (910 here)
normal_group = population_df[population_df['Status'] == 'Normal']

# Select subjects whose image quality control is A (930 here))
A_QC_group = population_df[population_df['quality_control'] == 'A']

# Keep only 'normal' subjects with images' quality control 'A', that is
# subjects ID whose images could be used to draw a new template:
# Get the intercept of indices of 'normal' subjects and of subjects with
# images' quality control 'A'
normal_subjects_id = np.intersect1d(normal_group.index.values,
                                    A_QC_group.index.values)


# Demographics of the selected normal subjects whose images are suitable for
# the creation of a new template
male_group = population_df.loc[normal_subjects_id][
                    population_df.loc[normal_subjects_id]
                    ['Gender de Feuil2'] == 'Male']
female_group = population_df.loc[normal_subjects_id][
                    population_df.loc[normal_subjects_id]
                    ['Gender de Feuil2'] == 'Female']

print "There are", male_group.shape[0], "male people of normal weight status."
print "There are", female_group.shape[0], "female people  of normal weight status."


## T1 images of neuroanatomy (after Gaser segmentation)
#IMAGEN_images = []
#for file in glob(os.path.join(GASER_VBM8_PATH, 'smwp1*.nii')):
#    IMAGEN_images.append(file)
#
## T1 images of neuroanatomy (after Gaser segmentation)
#img_id_list = []
#for i, image_id in enumerate(IMAGEN_images):
#    subjects_id = image_id[len(GASER_VBM8_PATH + '/smwp1'):
#                            -len('s301a1003.nii')]
#    if len(subjects_id) == 12:
#        subjects_id = int(subjects_id)
#    else:
#        subjects_id = int(subjects_id[:-1])
#
#    img_id_list.append(subjects_id)

# List of neuroanatomic images' pathes to use to draw a new template
template_img_list = []
for i, subject_id in enumerate(normal_subjects_id):
    image_id = GASER_VBM8_PATH + '/smwp1' + '%012d' % subject_id
    image_path = glob(image_id + '*.nii')
    template_img_list.append(image_id)


# Save the list of ID_images to be used to draw a new template as a .csv file
normal_template_images = pd.DataFrame.to_csv(pd.DataFrame(template_img_list),
                                os.path.join(TEMPLATE_PATH,
                                             'images_for_normal_template.txt'),
                                             index=False)