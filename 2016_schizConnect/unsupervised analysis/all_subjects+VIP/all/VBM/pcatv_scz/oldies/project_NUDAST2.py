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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy'

X_nudast = np.load(INPUT_DATA_X)
y_nudast = np.load(INPUT_DATA_y)

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)



hallu_saps7_scores = np.zeros(( X_nudast.shape[0]))
i=0
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    if len(curr)>0:
        most_recent_visit = curr.visit.unique()[-1]
        curr = curr[curr.visit == most_recent_visit]
        if curr[curr.question_id == "saps7"].empty == False:
                hallu_saps7_scores[i] = curr[curr.question_id == "saps7"].question_value.astype(np.int64).values
        else:
                hallu_saps7_scores[i] = np.nan
    i = i +1

hallu_saps7_scores_scz = hallu_saps7_scores[y_nudast == 1]

assert X_nudast.shape == (270, 125961)
X_nudast = X_nudast[:,2:]
assert X_nudast.shape == (270, 125959)

U_nudast, d = transform(V=components , X = X_nudast , n_components=components.shape[1], in_place=False)
assert U_nudast.shape == (270, 10)
U_nudast_scz = U_nudast[y_nudast==1,:]
assert U_nudast_scz.shape == (118, 10)




for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(hallu_saps7_scores_scz )==False)]
    y = hallu_saps7_scores_scz [np.array(np.isnan(hallu_saps7_scores_scz )==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],hallu_saps7_scores_scz ,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Hallu score P3")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/panss_pos"
for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    y = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],panss_pos_scz,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS POS")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/panss_neg"
for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
    y = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],panss_neg_scz,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS NEG")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/halluP3"
for i in range(10):
    plt.figure()
    x = U_nudast_scz[:,i][np.array(np.isnan(panss_scores_scz[:,2])==False)]
    y = panss_scores_scz[:,2][np.array(np.isnan(panss_scores_scz[:,2])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_nudast[y_nudast==1,i],panss_scores_scz[:,2],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Hallu score P3")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_nudast/boxplots"
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos
    df["score"] = U_nudast[:,i]
    df["dx"] = y_nudast
    T,pvalue = scipy.stats.ttest_ind(U_nudast[y_nudast==0,i],U_nudast[y_nudast==1,i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%i)
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/WAIS_COMPL_IM_TOT"
clinic_score = pop_vip["WAIS_COMPL_IM_TOT"]
for i in range(10):
    plt.figure()
    x = U_vip[:,i][np.array(np.isnan(clinic_score)==False)]
    y = clinic_score[np.array(np.isnan(clinic_score)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==0,i],clinic_score[y_vip==0],'o',label = "CTL")
    plt.plot(U_vip[y_vip==1,i],clinic_score[y_vip==1],'o',label = "SCZ")
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Total score for WAIS Picture Completion ")
    plt.legend()
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

###############################################################################

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