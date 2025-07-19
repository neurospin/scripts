# -*- coding: utf-8 -*-
"""
Created on Fri Jul  18  2025

"""

import os

# Manipulate data
import numpy as np
import pandas as pd
import itertools
# Statistics
import scipy.stats
import statsmodels.api as sm
#import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
#from statsmodels.stats.stattools import jarque_bera


################################################################################
# %% Input variables
clinical_vars_dict = dict(
    #QI = ['QI<85', '85-115', 'QI>115'],
    #Familiaux_psy=['actdpsy1', 'atcd_depression1', 'atcd_TH1', 'atcd_psychose1', 'atcd_TND1'],
    TND_DSM_V = ['DSM_MOT', 'DSM_TSA', 'DSM_TDAH', 'DSM_Tr App'],
    other = ['Catatonie'],
    TEMPSA= ['TEMPSA-C', 'TEMPSA-D', 'TEMPSA-I', 'TEMPSA-H', 'TEMPSA-A'],
    PQ16= ['PQ16-T', 'PQ16-A'],
    CDI=['CDI'],
    ASQ= ['ASQtot'],
    Atcd_trauma=['atcd_trauma']
)

clinical_vars = [v for set in clinical_vars_dict.values() for v in set]  # Flatten the list of input variables


################################################################################
# %% Config

config = dict(
    # Set the working directory
    working_directory='/home/ed203246/git/scripts/2025_NeuroLithe',
    # Set the path to the data file
    data_file='data/NeuroLithe_V1707.xlsx',
    response_with_PSP=True,  # If True, response is defined with PSP_FONCTIONNEMENT >= 70
    # Set the path to save results
    output_models='models/',
    clinical_vars=clinical_vars,
    demo_vars = ['AGE', 'Sexe'],
    metrics=["accuracy", "balanced_accuracy", "roc_auc"]

)

# Set Working Directory
os.chdir(config['working_directory'])


################################################################################
# %% Load data
nrows = 61 - 3
data = pd.read_excel(config['data_file'], sheet_name='Database', skiprows=2, nrows=nrows)
assert data.Patient.iloc[-1] == 'NEUROLITHE_058'
data.dtypes
# Display first few rows of the dataset
print(data.head())

# Response

if config['response_with_PSP']:
    response = \
        (data.rehospi_T2 == 0) & (data.Scolarite_T2 == 1) & (data.PSP_FONCTIONNEMENT >= 70)

else:
    response = \
        (data.rehospi_T2 == 0) & (data.Scolarite_T2 == 1)

response = response.astype(int)
data['response'] = response

config['demo_vars']

# Check that all input variables are numeric
print(data[config['demo_vars'] + config['clinical_vars'] +  ['response']].dtypes)
assert len(data[config['demo_vars'] + config['clinical_vars'] +  ['response']].select_dtypes(include=np.number).columns) ==\
    len(config['demo_vars'] + config['clinical_vars'] +  ['response'])  


################################################################################

# %% Utils
# ========

