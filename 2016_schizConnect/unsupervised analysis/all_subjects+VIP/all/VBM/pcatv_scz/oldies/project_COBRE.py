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
from sklearn import svm, metrics, linear_model

################
# Input/Output #
################

components = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/vbm_pcatv_all+VIP_scz/results/0/struct_pca_0.1_0.1_0.1/components.npz")["arr_0"]


assert components.shape == (125959, 10)


#NUDAST
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_COBRE_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/y.npy'


X_cobre = np.load(INPUT_DATA_X)
y_cobre = np.load(INPUT_DATA_y)

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
age = pop["age"].values

PANSS_MAP = {"Absent": 1, "Minimal": 2, "Mild": 3, "Moderate": 4, "Moderate severe": 5, "Severe": 6, "Extreme": 7,\
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
clinic["question_value_panss_scale"] = clinic["question_value"].map(PANSS_MAP)


panss_scores = np.zeros((164,30))
CNP = np.zeros((164,105))
i=0
for s in pop.subjectid:

    curr = clinic[clinic.subjectid ==s]
    for k in range(1,31):
        if curr[curr.question_id == "FIPAN_%s"%k].empty == False:
            panss_scores[i,k-1] = curr[curr.question_id == "FIPAN_%s"%k].question_value_panss_scale.values
        else:
            panss_scores[i,k-1] = np.nan
            if(y_cobre[i]==1.0):
                print(s)
    i = i +1

panss_pos = np.sum(panss_scores[:,:7],axis=1)
panss_neg = np.sum(panss_scores[:,7:14],axis=1)
panss_scores_scz = panss_scores[y_cobre==1,:]
panss_pos_scz = panss_pos[y_cobre==1,]
panss_neg_scz = panss_neg[y_cobre==1,]


CNP = np.zeros((164,100))
i=0
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for k in range(1,100):
        if curr[curr.question_id == "CNP_%s"%k].empty == False:
            CNP[i,k] = curr[curr.question_id == "CNP_%s"%k].question_value.astype(np.float).values[0]
        else:
            CNP[i,k] = np.nan
    i = i +1

CNP_scz = CNP[y_cobre==1,:]


plt.plot(panss_pos_scz,panss_neg_scz,'o')
plt.xlabel("PANSS positive")
plt.ylabel("PANSS negative")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/projection_cobre/panss.png")

assert X_cobre.shape == (164, 125961)
X_cobre = X_cobre[:,2:]
assert X_cobre.shape == (164, 125959)

U_cobre, d = transform(V=components , X = X_cobre , n_components=components.shape[1], in_place=False)
assert U_cobre.shape == (164, 10)
U_cobre_scz = U_cobre[y_cobre==1,:]
assert U_cobre_scz.shape == (77, 10)
U_cobre_con = U_cobre[y_cobre==0,:]

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/data/U_cobre.npy",U_cobre)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/data/U_cobre_con.npy",U_cobre_con)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/data/U_cobre_scz.npy",U_cobre_scz)


#Correct effect of age
##############################################################################
# function to correct scores for age using controls
U_patients_corr = np.zeros(U_cobre_scz.shape)
for i in range(10):
    regr = linear_model.LinearRegression()
    #regr.fit(age[y_vip==1].reshape(-1, 1),U_vip_scz[:,i])
    regr.fit(age[y_cobre==0].reshape(-1, 1),U_cobre_con[:,i])
    #regr.fit(age.reshape(-1, 1),U_vip[:,i])
    score_pred = regr.predict(age[y_cobre==1].reshape(-1, 1))
    scores_corrected = U_cobre_scz[:,i] - score_pred
    U_patients_corr[:,i] =  scores_corrected

##############################################################################
U_cobre_scz = U_patients_corr

import statsmodels.api as sm
from statsmodels.formula.api import ols

for i in range(10):
    df = pd.DataFrame()
    df["age"] = age[y_cobre==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["panss_pos"] = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_cobre_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_pos +age",data = df).fit()
    print(i)
    print(mod.summary())



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/panss_pos"
for i in range(10):
    plt.figure()
    x = U_cobre_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    y = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_cobre[y_cobre==1,i],panss_pos_scz,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS POS")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/panss_neg"
for i in range(10):
    plt.figure()
    x = U_cobre_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
    y = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_cobre[y_cobre==1,i],panss_neg_scz,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS NEG")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/halluP3"
for i in range(10):
    plt.figure()
    x = U_cobre_scz[:,i][np.array(np.isnan(panss_scores_scz[:,2])==False)]
    y = panss_scores_scz[:,2][np.array(np.isnan(panss_scores_scz[:,2])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_cobre[y_cobre==1,i],panss_scores_scz[:,2],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Hallu score P3")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

for i in range(10):
    plt.figure()
    x = U_cobre_scz[:,i][np.array(np.isnan( CNP_scz[:,99])==False)]
    y =  CNP_scz [:,99][np.array(np.isnan( CNP_scz[:,99])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_cobre[y_cobre==1,i], CNP_scz [:,99],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Hallu score P3")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))






################################################################################
output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_cobre/boxplots"
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos
    df["score"] = U_cobre[:,i]
    df["dx"] = y_cobre
    T,pvalue = scipy.stats.ttest_ind(U_cobre[y_cobre==0,i],U_cobre[y_cobre==1,i])
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