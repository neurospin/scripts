# -*- coding: utf-8 -*-
"""
Created on Fri Jul  18  2025


PROMPT
**Profile** I am a Research Director and Professor of Machine Learning, Head of a Laboratory for Brain Imaging and Data Science. My research focuses on developing and applying machine learning and statistical models to identify neural signatures predictive of clinical trajectories in psychiatric disorders.**Context** I am drafting a paper entitled "Predictors of Lithium Response in Adolescents with Persistent Mood Symptoms: A longitudinal Study". This paper aims to be submitted to "Biological Psychiatry". I am analysing the data from the Neurolithe study, a monocentric, longitudinal cohort study of 58 adolescents aged 12–17 years
with persistent mood symptoms, with or without psychotic features, treated with lithium.

The primary aim was to identify clinical and neurodevelopmental predictors of lithium response.
The input are the Neurodevelopmental and clinical profiles that were assessed at baseline using self-report
questionnaires and clinical evaluations. Lisit of input variables:
DSM_TDAH
TEMPSA-C
PQ16-A
Catatonie
DSM_MOT
ASQtot
TEMPSA-H
PQ16-T
CDI
TEMPSA-D
DSM_Tr App
TEMPSA-I
atcd_trauma
DSM_TSA
TEMPSA-A

The output is a binary variable of lithium response, assessed after a minimum of 6-month follow-up and defined by functional and clinical criteria, including a PSP score ≥70, absence of hospitalization, and school reintegration.

I use a logistic regression model with L2 regularization to predict lithium response from the input variables,
and I want to evaluate the performance of the model using repeated cross-validation and permutation testing.
I also want to compute feature importance and statistical significance of the features.

Let Xdf be the dataframe of input variables, and y be the binary variable of lithium response.
Let cv_test = StratifiedKFold(n_splits=5, shuffle=True, random_state=8) be the stratified cross-validation scheme for testing.
Let model = make_pipeline([preprocessing.StandardScaler(), GridSearchCV(lm.LogisticRegression(fit_intercept=False, class_weight='balanced'), {'C': 10. ** np.arange(-3, 1)}, cv=cv_val, n_jobs=5, scoring='accuracy')])
be the predictive model for classification problem.

**Instructions** 
Help me to design the explainability analysis in python:
(i) Propose an exploratory data analysis to understand the correlation structure between features and their relationship with the response variable, including appropriate plots and statistics;
(ii) Classical feature importance analysis is prone to multicolinarity issues, which can lead to misleading interpretations. Propose a method to determine the feature importance in the predictive model


, and to improve the interpretation by organizing features into components (e.g. using PCA or clustering), and to visualize the feature importance in the context of these components.

propose a method to determine the feature importance in the predictive model, and to improve the interpretation by organizing features into components (e.g. using PCA or clustering), and to visualize the feature

propose plot to analyse the correlation structure between features
(ii) propose plot to analyse the feature importance in the predictive model

(iii) 
(i) determine the feature importance in the predictive model ;
(ii) to improve the interpretation by organizing features into components 
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


