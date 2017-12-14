#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:42:03 2017

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import os
import numpy as np
import pandas as pd
import nibabel as nb
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
################
# Input/Output #
################

components = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1/components.npz")["arr_0"]


assert components.shape == (125959, 10)


#NUDAST
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy'


clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
pop["SIPSP"] =  "NaN"
pop["SIPSN"] =  "NaN"
pop["SCID"] =  "NaN"

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr = curr[curr.visit == most_recent_visit]
    current_SAPS = curr[curr.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.int64).values
    current_SANS = curr[curr.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.int64).values
    current_BPRS = curr[curr.assessment_description == "Brief Psychiatric Rating Scale"].question_value.astype(np.int64).values
    current_SIPSP = curr[curr.assessment_description == "Structured Interview for Prodromal Syndromes Summary"].question_value.astype(np.int64).values
    current_SCID = curr[curr.assessment_description == "Scid Summary"].question_value.astype(np.int64).values
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
    if len(current_SAPS) != 0:
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()
    if len(current_BPRS) != 0:
        pop.loc[pop.subjectid ==s,"BPRS"] = current_BPRS.sum()
    if len(current_SIPSP) != 0:
        pop.loc[pop.subjectid ==s,"SIPSP"] = current_SIPSP.sum()
    if len(current_SCID) != 0:
        pop.loc[pop.subjectid ==s,"SCID"] = current_SCID.sum()


panss_pos =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
panss_neg =  pop[pop.dx_num ==1].SANS.astype(np.float).values
BPRS_scores =  pop[pop.dx_num ==1].BPRS.astype(np.float).values
SIPSP_scores =  pop[pop.dx_num ==1].SIPSP.astype(np.float).values
SCID_scores =  pop[pop.dx_num ==1].SCID.astype(np.float).values


panss_comp = panss_neg - panss_pos


X_nudast = np.load(INPUT_DATA_X)
y_nudast = np.load(INPUT_DATA_y)

assert X_nudast.shape == (270, 125961)
X_nudast = X_nudast[:,2:]
assert X_nudast.shape == (270, 125959)

U_nudast, d = transform(V=components , X = X_nudast , n_components=components.shape[1], in_place=False)
assert U_nudast.shape == (270, 10)
U_nudast_scz = U_nudast[y_nudast==1,:]
assert U_nudast_scz.shape == (118, 10)


plt.plot(panss_pos,panss_neg,'o')
plt.xlabel("PANSS positive")
plt.ylabel("PANSS negative")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/projection_nudast/panss.png")


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/panss_neg"
for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(panss_neg)==False)]
    y = panss_neg[np.array(np.isnan(panss_neg)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],panss_neg,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS NEG")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/panss_comp"
for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(panss_comp)==False)]
    y = panss_comp[np.array(np.isnan(panss_comp)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],panss_comp,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS NEG")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(SCID_scores)==False)]
    y = SCID_scores[np.array(np.isnan(SCID_scores)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],SCID_scores,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("SCID_scores")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))



################################################################################
panss_neg_scz = panss_neg[y_nudast==1]
panss_pos_scz = panss_pos[y_nudast==1]

for i in range(10):
    plt.figure()
    plt.plot(U_nudast_scz[np.array(panss_neg_scz<30),1],U_nudast_scz[np.array(panss_neg_scz<30),i],'o')
    plt.plot(U_nudast_scz[np.array(panss_neg_scz>25),1],U_nudast_scz[np.array(panss_neg_scz>25),i],'o')

for i in range(10):
    plt.figure()
    plt.plot(U_nudast_scz[np.array(panss_pos_scz<20),1],U_nudast_scz[np.array(panss_pos_scz<20),i],'o')
    plt.plot(U_nudast_scz[np.array(panss_pos_scz>20),1],U_nudast_scz[np.array(panss_pos_scz>20),i],'o')

################################################################################
################################################################################

def transform(V, X, n_components, in_place=False):
    """ Project a (new) dataset onto the components.
    Return the projected data and the associated d.
    We have to recompute U and d because the argument may not have the same
    number of lines.
    The argument must have the same number of columns than the datset used
    to fit the estimator.
    """
    Xk = X
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