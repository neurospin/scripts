#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:15:17 2017

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


plt.plot(SAPS_scores,SANS_scores,'o')
corr,p = scipy.stats.pearsonr(SAPS_scores,SANS_scores)
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel("SAPS score")
plt.ylabel("SANS score")

age =  pop.age.values
sex =  pop.sex_num.values

age_scz =  pop[pop.dx_num ==1].age.values
sex_scz =  pop[pop.dx_num ==1].sex_num.values

comp_scz = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","components.npz"))['arr_0']


U_controls, d = transform(V=comp_scz, X=X[y==0,3:], n_components=comp_scz.shape[1], in_place=False)
U_patients, d = transform(V=comp_scz, X=X[y==1,3:], n_components=comp_scz.shape[1], in_place=False)
U = np.vstack((U_controls,U_patients))



INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
pop = pd.read_csv(INPUT_POPULATION)
age_controls =  pop[pop.dx_num ==0].age.values
sex_controls =  pop[pop.dx_num ==0].sex_num.values

age= age.reshape(270,1)
age_scz = age_scz.reshape(118,1)
age_controls = age_controls.reshape(152,1)
sex = sex.reshape(270,1)
sex_scz = sex_scz.reshape(118,1)
sex_controls = sex_controls.reshape(152,1)

age_sex_controls = np.concatenate((age_controls,sex_controls),axis=1)
age_sex_scz = np.concatenate((age_scz,sex_scz),axis=1)
age_sex = np.concatenate((age,sex),axis=1)


U_patients_corr = np.zeros(U_patients.shape)
# function to correct scores for age and gender using controls
for i in range(10):
    score_patients = U_patients[:,i].reshape(118,1)
    score_controls = U_controls[:,i].reshape(152,1)
    score_all = U[:,i].reshape(270,1)
    from sklearn import svm, metrics, linear_model
    regr = linear_model.LinearRegression()
    regr.fit(age_sex_controls,score_controls)
    #regr.fit(age_sex_scz,score_patients)
    #regr.fit(age_sex,score_all)
    print('Coefficients: \n', regr.coef_)
    score_pred = regr.predict(age_sex_scz)
    scores_corrected = U_patients[:,i] - score_pred[:,0]
    U_patients_corr[:,i] =  scores_corrected
    pearsonr(age_scz[:,0],scores_corrected)
    pearsonr(age_scz[:,0],U_patients[:,i])
    pearsonr(U_patients[:,i],SAPS_scores)
    print ("comp")
    print (i)
    print(pearsonr(scores_corrected,SAPS_scores))
    print(pearsonr(scores_corrected,SANS_scores))

    #comp 3
###############################################################################
plt.figure()
corr,p = pearsonr(SAPS_scores,U_patients_corr[:,3],)
plt.plot(U_patients_corr[:,3],SAPS_scores,'o')
plt.xlabel('Score on component 3')
plt.ylabel('SAPS score, corrected for age and gender')
plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)


plt.figure()
corr,p = pearsonr(SANS_scores,U_patients_corr[:,2],)
plt.plot(U_patients_corr[:,2],SANS_scores,'o')
plt.xlabel('Score on component 2')
plt.ylabel('SANS score, corrected for age and gender')
plt.title("Pearson's correlation = %.02f, p = %.01e" % (corr,p),fontsize=12)
###############################################################################

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_corr[:,i],SANS_scores)
    axs[i].plot(U_patients_corr[:,i],SANS_scores,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score, corr for age/gender')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
    plt.tight_layout()

fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_corr[:,i],SAPS_scores)
    axs[i].plot(U_patients_corr[:,i],SAPS_scores,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score, corr for age/gender')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
    plt.tight_layout()

def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = check_arrays(X)
    if not in_place:
        Xk = Xk.copy()
    n, p = Xk.shape
    if p != V.shape[0]:
        raise ValueError(
                    "The argument must have the same number of columns "
                    "than the datset used to fit the estimator.")
    U = np.zeros((n, n_components))
    d = np.zeros((n_components, ))
    for k in range(n_components):
        # Project on component j
        vk = V[:, k].reshape(-1, 1)
        uk = np.dot(X, vk)
        uk /= np.linalg.norm(uk)
        U[:, k] = uk[:, 0]
        dk = np.dot(uk.T, np.dot(Xk, vk))
        d[k] = dk
        # Residualize
        Xk -= dk * np.dot(uk, vk.T)
    return U, d