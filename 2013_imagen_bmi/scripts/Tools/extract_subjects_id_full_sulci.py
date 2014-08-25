# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:21:43 2014

@author: hl237680

This script generates the list of IDs of subjects who passed the quality
control on sulci data.
Then, it creates a .csv file giving gender, age in years, BMI and weight
status for the 978 subjects who passed the quality control on sulci data.

INPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_df_qc.csv"

OUTPUT: .csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv"

    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/full_sulci_clinics.csv"

Only 978 subjects among the 1265 for who we have both genetic and neuroimaging
data (masked_images) have passed the quality control step.
"""

import os
import numpy as np
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SHFJ_DATA_PATH = os.path.join(CLINIC_DATA_PATH, 'source_SHFJ')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')


# Sulci features
sulci_df_qc = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                  'sulci_df_qc.csv'),
                                      sep=',',
                                      index_col=0)

# Consider subjects for whom we have neuroimaging and genetic data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH,
                                         'subjects_id.csv'),
                            dtype=None,
                            delimiter=',',
                            skip_header=1)

# Keep only subjects for whom we have neuroimaging, genetics and sulcal
# data: get the intercept of indices of subjects for whom we have
# neuroimaging and genetic data, but also sulci features
subjects_index = np.intersect1d(subjects_id, sulci_df_qc.index.values)

# Convert subjects_index numpy.array to list
subjects_index_list = subjects_index.tolist()

# Save list of subjects ID as .csv file
subjects_id_full_sulci = pd.DataFrame.to_csv(pd.DataFrame
                                                (subjects_index_list,
                                                 columns=['subject_id']),
                                        os.path.join(FULL_SULCI_PATH,
                                            'subjects_id_full_sulci.csv'),
                                        index=False)


# File population.csv containing all clinical data for IMAGEN subjects
population_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                 'population.csv'),
                                      sep=',',
                                      index_col=0)

# Parameters of interest
colnames = ['Gender de Feuil2', 'ageannees', 'BMI', 'Status']

# Dataframe containing data of interest covariates for the selected subjects
full_sulci_clinic_df = pd.DataFrame(population_df,
                                    index=subjects_index_list,
                                    columns=colnames)

full_sulci_clinics = pd.DataFrame.to_csv(full_sulci_clinic_df,
                                         os.path.join(FULL_SULCI_PATH,
                                                    'full_sulci_clinics.csv'))


# Demographics
insuff_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Insuff']
normal_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Normal']
overweight_group = full_sulci_clinic_df[full_sulci_clinic_df
                                            ['Status'] == 'Overweight']
obese_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Obese']

male_group = full_sulci_clinic_df[full_sulci_clinic_df
                                            ['Gender de Feuil2'] == 'Male']
female_group = full_sulci_clinic_df[full_sulci_clinic_df
                                            ['Gender de Feuil2'] == 'Female']

print "There are ", insuff_group.shape[0], "thick people."
print "There are ", normal_group.shape[0], "normal people."
print "There are ", overweight_group.shape[0], "overweight people."
print "There are ", obese_group.shape[0], "obese people."

print "There are ", male_group.shape[0], "male people."
print "There are ", female_group.shape[0], "female people."


# Demographics of all subjects for whom we have both neuroimaging and genetic
# data
clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                  'population.csv'),
                                      sep=',',
                                      index_col=0)

insuff_group_orig = clinical_df[clinical_df['Status'] == 'Insuff']
normal_group_orig = clinical_df[clinical_df['Status'] == 'Normal']
overweight_group_orig = clinical_df[clinical_df['Status'] == 'Overweight']
obese_group_orig = clinical_df[clinical_df['Status'] == 'Obese']

male_group_orig = clinical_df[clinical_df['Gender de Feuil2'] == 'Male']
female_group_orig = clinical_df[clinical_df['Gender de Feuil2'] == 'Female']

print "There are ", insuff_group_orig.shape[0], "thick people in original data."
print "There are ", normal_group_orig.shape[0], "normal people in original data."
print "There are ", overweight_group_orig.shape[0], "overweight people in original data."
print "There are ", obese_group_orig.shape[0], "obese people in original data."

print "There are ", male_group_orig.shape[0], "male people in original data."
print "There are ", female_group_orig.shape[0], "female people in original data."