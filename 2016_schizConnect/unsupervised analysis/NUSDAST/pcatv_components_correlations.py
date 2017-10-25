#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:05:18 2017

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

age_scz =  pop[pop.dx_num ==1].age.values
sex_scz =  pop[pop.dx_num ==1].sex_num.values

comp_scz = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","components.npz"))['arr_0']


U_controls, d = transform(V=comp_scz, X=X[y==0,3:], n_components=comp_scz.shape[1], in_place=False)
U_patients, d = transform(V=comp_scz, X=X[y==1,3:], n_components=comp_scz.shape[1], in_place=False)



INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
pop = pd.read_csv(INPUT_POPULATION)
age_controls =  pop[pop.dx_num ==0].age.values
sex_controls =  pop[pop.dx_num ==0].sex_num.values


age_scz = age_scz.reshape(118,1)
age_controls = age_controls.reshape(152,1)
sex_scz = sex_scz.reshape(118,1)
sex_controls = sex_controls.reshape(152,1)



U_patients_corr = np.zeros(U_patients.shape)
# function to correct scores for age using controls
for i in range(10):
    score_patients = U_patients[:,i].reshape(118,1)
    score_controls = U_controls[:,i].reshape(152,1)
#    plt.plot(age_controls,score_controls,'o')
#    plt.plot(age_scz,score_patients,'o')
    from sklearn import svm, metrics, linear_model
    regr = linear_model.LinearRegression()
    regr.fit(age_controls,score_controls)
    print('Coefficients: \n', regr.coef_)
    # Plot outputs
#    plt.scatter(age_controls, score_controls,color='black')
#    plt.plot(age_controls, regr.predict(age_controls), color='blue',linewidth=3)
#    plt.xlabel("age")
#    plt.xlabel("score_0")
#    plt.show()
#
#    plt.scatter(age_scz, score_patients,color='black')
#    plt.plot(age_scz, regr.predict(age_scz), color='blue',linewidth=3)
#    plt.xlabel("age")
#    plt.xlabel("score_0")
#    plt.show()
    score_pred = regr.predict(age_scz)
    scores_corrected = U_patients[:,i] - score_pred[:,0]
    U_patients_corr[:,i] =  scores_corrected
#    plt.plot(age_scz,scores_corrected,'o')
#    plt.plot(age_scz,U_patients[:,i],'o')
    pearsonr(age_scz[:,0],scores_corrected)
    pearsonr(age_scz[:,0],U_patients[:,i])
    pearsonr(U_patients[:,i],SAPS_scores)
    print ("comp")
    print (i)
    print(pearsonr(scores_corrected,SAPS_scores))
    print(pearsonr(scores_corrected,SANS_scores))
#    plt.plot(U_patients[:,i],SAPS_scores,'o')
#    plt.plot(scores_corrected,SAPS_scores,'o')

#SAPS, MEN ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_corr[(sex_scz==0).ravel(),i],SAPS_scores[(sex_scz==0).ravel()])
    axs[i].plot(U_patients_corr[(sex_scz==0).ravel(),i],SAPS_scores[(sex_scz==0).ravel()],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr MALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score, corrected for age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
plt.tight_layout()

#SAPS, WOMEN ONLY
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_corr[(sex_scz==1).ravel(),i],SAPS_scores[(sex_scz==1).ravel()])
    axs[i].plot(U_patients_corr[(sex_scz==1).ravel(),i],SAPS_scores[(sex_scz==1).ravel()],'o', markersize = 4,label = "male")
    axs[i].set_title("Pearson corr FEMALE ONLY = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score, corrected for age')
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
    axs[i].set_ylabel('SAPS score, adjusted for age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SAPSS_age_corrected.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SAPSS_age_corrected.png")


fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_corr[:,i],SANS_scores)
    axs[i].plot(U_patients_corr[:,i],SANS_scores,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SANS score, , adjusted for age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SANSS_age_corrected.pdf")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp/correlation_SANSS_age_corrected.png")


fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients[:,i],SANS_scores[:])
    axs[i].plot(U_patients[sex_scz==1,i],SANS_scores[sex_scz==1],'o', markersize = 4)
    axs[i].plot(U_patients[sex_scz==0,i],SANS_scores[sex_scz==0],'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SANS score, , adjusted for age')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))



for i in range(10):
    print(pearsonr(U_patients[sex_scz==1,i],SANS_scores[sex_scz==1]))
    print(pearsonr(U_patients[sex_scz==0,i],SANS_scores[sex_scz==0]))

for i in range(10):
    print(pearsonr(U_patients[sex_scz==1,i],SAPS_scores[sex_scz==1]))
    print(pearsonr(U_patients[sex_scz==0,i],SAPS_scores[sex_scz==0]))


################################################################################
###############################################################################
#IDEA :  Take components yielded by all schizco patients
# Find Projections of all NUDAST in this basis.
#check correlations

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel
import pandas as pd
import nibabel as nib
import json
import nilearn
from nilearn import plotting
from nilearn import image
import array_utils

INPUT_MASK = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/mask.nii'

babel_mask  = nib.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
#############################################################################
#SCZ only
#############################################################################

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1"

comp = np.load(os.path.join(WD,"components.npz"))['arr_0']

N_COMP = 10

for i in range(comp.shape[1]):
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] =comp[:,i]
    out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
    filename = os.path.join(WD,"comp_%s.nii.gz") % (i)
    out_im.to_filename(filename)
    comp_data = nibabel.load(filename).get_data()
    comp_t,t = array_utils.arr_threshold_from_norm2_ratio(comp_data, .99)
    nilearn.plotting.plot_glass_brain(filename,colorbar=True,plot_abs=False,threshold = t)
    print (i)
    print (t)




INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/results/pcatv_scz/5_folds_NUDAST_10comp"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/data/X_scz_only.npy'
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")
# Compute clinical Scores

X = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/X_scz.npy")

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
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
    if len(current_SAPS) != 0:
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()


SAPS_scores =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_scores =  pop[pop.dx_num ==1].SANS.astype(np.float).values

plt.plot(SAPS_scores,SANS_scores,'o')
corr,p = pearsonr(SAPS_scores,SANS_scores)
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel("SAPS score")
plt.ylabel("SANS score")

age_scz =  pop[pop.dx_num ==1].age.values

U_patients, d = transform(V=comp, X=X[:,4:], n_components=comp.shape[1], in_place=False)

#sum(X[:,3] == 0.49844999502433401) correspond to NUDAST subject
index_NUDAST = np.where(X[:,3] == 0.49844999502433401)

U_patients_NUDAST = U_patients[index_NUDAST,:].reshape(118,10)


 # PLOT ALL CORRELATIONW WITH SAPS
fig, axs = plt.subplots(2,5, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = 1.0, wspace=.3)
axs = axs.ravel()
for i in range(10):
    corr,p = pearsonr(U_patients_NUDAST[:,i],SAPS_scores)
    axs[i].plot(U_patients_NUDAST[:,i],SAPS_scores,'o', markersize = 4)
    axs[i].set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    axs[i].xaxis.set_ticks(np.arange(-0.3,0.4,0.2))
    axs[i].set_xlabel('Score on component %s' %(i+1))
    axs[i].set_ylabel('SAPS score')
    axs[i].yaxis.set_ticks(np.arange(10,80,10))
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
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