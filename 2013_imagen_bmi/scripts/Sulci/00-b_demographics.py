# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 10:24:15 2014

@author: hl237680

This script gives an insight of the distribution of ponderal status among
subjects who passed the quality control for the study on sulci.

INPUT:
- Clinical data of IMAGEN subjects:
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"
- Sulci features after quality control:
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_df_qc.csv"

OUTPUT:
    returns the number of subjects, among those who passed the quality control
    for the study on sulci, who are insuff, normal, overweight or obese.

"""

import os
import numpy as np
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')


# Sulci features
sulci_df_qc = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                                  'sulci_df_qc.csv'),
                                      sep=',',
                                      index_col=0)

# Clinical data
clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                  'population.csv'),
                                     index_col=0)

# Subjects ID for sulci study
subjects_id = np.intersect1d(sulci_df_qc.index.values,
                             clinical_df.index.values)

# Clinical data for the 745 subjects who passed the quality control on sulci
pop_sulci = clinical_df.loc[subjects_id]

print '###################'
print '# Group selection #'
print '###################'

insuf_group = pop_sulci[pop_sulci['Status'] == 'Insuff']
print insuf_group.shape[0], "insuff."
normal_group = pop_sulci[pop_sulci['Status'] == 'Normal']
print normal_group.shape[0], "normal."
overweight_group = pop_sulci[pop_sulci['Status'] == 'Overweight']
print overweight_group.shape[0], "overweight."
obese_group = pop_sulci[pop_sulci['Status'] == 'Obese']
print obese_group.shape[0], "obese."