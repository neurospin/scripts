#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:43:13 2017

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
import scipy.stats
from scipy.stats import pearsonr
###############################################################################
# SCZ ONLY
###############################################################################


#LOAD SCORES
###############################################################################
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X_scz_only.npy'
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")
# Compute clinical Scores

X = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_all/5_folds_NUDAST_all/X.npy")
y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/y.npy")

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr = curr[curr.visit == most_recent_visit]
    current_SAPS = curr[curr.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.int64).values
    current_SANS = curr[curr.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.int64).values
    current_BPRS = curr[curr.assessment_description == "Brief Psychiatric Rating Scale"].question_value.astype(np.int64).values
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
    if len(current_SAPS) != 0:
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()
    if len(current_BPRS) != 0:
        pop.loc[pop.subjectid ==s,"BPRS"] = current_BPRS.sum()


SAPS_scores =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_scores =  pop[pop.dx_num ==1].SANS.astype(np.float).values
BPRS_scores =  pop[pop.dx_num ==1].BPRS.astype(np.float).values

age_scz =  pop[pop.dx_num ==1].age.values
sex_scz =  pop[pop.dx_num ==1].sex_num.values

#controls indormation
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
pop = pd.read_csv(INPUT_POPULATION)
age_controls =  pop[pop.dx_num ==0].age.values
sex_controls =  pop[pop.dx_num ==0].sex_num.values


#COMPUTE PROJECTION
###############################################################################
comp_scz = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","components.npz"))['arr_0']
U_controls, d = transform(V=comp_scz, X=X[y==0,3:], n_components=comp_scz.shape[1], in_place=False)
U_patients, d = transform(V=comp_scz, X=X[y==1,3:], n_components=comp_scz.shape[1], in_place=False)

age_scz = age_scz.reshape(118,1)
age_controls = age_controls.reshape(152,1)

sex_scz = sex_scz.reshape(118,1)
sex_controls = sex_controls.reshape(152,1)


#Adjust for age
##############################################################################

U_patients_corr = np.zeros(U_patients.shape)
# function to correct scores for age using controls
for i in range(10):
    score_patients = U_patients[:,i].reshape(118,1)
    score_controls = U_controls[:,i].reshape(152,1)
    from sklearn import svm, metrics, linear_model
    regr = linear_model.LinearRegression()
    regr.fit(age_controls,score_controls)
    score_pred = regr.predict(age_scz)
    scores_corrected = U_patients[:,i] - score_pred[:,0]
    U_patients_corr[:,i] =  scores_corrected
    print ("comp")
    print (i)
    print(pearsonr(scores_corrected,SAPS_scores))
    print(pearsonr(scores_corrected,SANS_scores))
#    plt.plot(U_patients[:,i],SAPS_scores,'o')
#    plt.plot(scores_corrected,SAPS_scores,'o')

#Adjust for age
##############################################################################

U_patients_corr = np.zeros(U_patients_corr.shape)
# function to correct scores for age using controls
for i in range(10):
    score_patients = U_patients_corr[:,i].reshape(118,1)
    score_controls = U_controls[:,i].reshape(152,1)
    from sklearn import svm, metrics, linear_model
    regr = linear_model.LinearRegression()
    regr.fit(sex_controls,score_controls)
    score_pred = regr.predict(sex_scz)
    scores_corrected = U_patients_corr[:,i] - score_pred[:,0]
    U_patients_corr[:,i] =  scores_corrected
    print ("comp")
    print (i)
    print(pearsonr(scores_corrected,SAPS_scores))
    print(pearsonr(scores_corrected,SANS_scores))
#    plt.plot(U_patients[:,i],SAPS_scores,'o')
#    plt.plot(scores_corrected,SAPS_scores,'o')
