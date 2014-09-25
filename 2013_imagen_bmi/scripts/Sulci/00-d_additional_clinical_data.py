# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:29:40 2014

@author: hl237680

Generation of a .csv file containing clinical data with additional parameters
such as gestational duration and socio-economic factor for IMAGEN subjects
who passed the QC on sulci.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/more_clinics.csv"
    .csv file containing both traditional clinical data but also additional
    parameters such as gestational duration and socio-economic factor

- "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_depthMax_df.csv"
    sulci maximal depth after quality control

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/more_clinics_745sulci.csv"
    complete clinical data for subjects who passed the quality control on
    sulci
"""


import os
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SHFJ_DATA_PATH = os.path.join(CLINIC_DATA_PATH, 'source_SHFJ')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')


# Clinical data
sup_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                             'more_clinics.csv'),
                                index_col=0)

# Sulci maximal depth
sulci_depthMax_df = pd.io.parsers.read_csv(os.path.join(QC_PATH,
                                            'sulci_depthMax_df.csv'),
                                           sep=',',
                                           index_col=0)

# Keep only subjects whose sulci have been robustly segmented
sup_df = sup_df.loc[sulci_depthMax_df.index.values]

# Write dataframe into a .csv file
sup_df.to_csv(os.path.join(CLINIC_DATA_PATH,
                           'more_clinics_745sulci.csv'))