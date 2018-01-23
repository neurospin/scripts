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

pop_vip = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv")
age = pop_vip["age"].values
panss_neg = pop_vip["PANSS_NEGATIVE"]
panss_pos = pop_vip["PANSS_POSITIVE"]
panss_galp = pop_vip["PANSS_GALPSYCHOPAT"]
panss_comp = pop_vip["PANSS_COMPOSITE"]
cdss = pop_vip["CDSS_Total"]
fast = pop_vip["FAST_TOT"]
hallu_panssP3 = pop_vip["PANSS_P3"]
dose = np.load("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/dose_ongoing_treatment.npy")



y_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/y.npy")
X_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/X.npy")
assert X_vip.shape == (92, 125961)
X_vip = X_vip[:,2:]
U_vip, d = transform(V=components , X = X_vip , n_components=components.shape[1], in_place=False)
assert U_vip.shape == (92, 10)
U_vip_scz = U_vip[y_vip==1,:]
U_vip_con = U_vip[y_vip==0,:]

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/U_vip.npy",U_vip)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/U_vip_scz.npy",U_vip_scz)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/U_vip_con.npy",U_vip_con)

panss_pos_scz = panss_pos[y_vip==1]
panss_neg_scz = panss_neg[y_vip==1]
panss_comp_scz = panss_comp[y_vip==1]
hallu_panssP3_scz  = hallu_panssP3[y_vip==1]


df = pd.DataFrame()
df["panss_pos"] = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
df["U"] = U_vip_scz[:,9][np.array(np.isnan(panss_pos_scz)==False)]

import statsmodels.api as sm
from statsmodels.formula.api import ols
mod = ols("U ~ panss_pos",data = df).fit()
mod.summary()

mod = ols("U ~ panss_pos +age",data = df).fit()
mod.summary()


plt.plot(panss_pos_scz,panss_neg_scz,'o')
plt.xlabel("PANSS positive")
plt.ylabel("PANSS negative")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/projection_vip/panss.png")



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/age"
for i in range(10):
    plt.figure()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    x0 = age[y_vip==0]
    y0 = U_vip[y_vip==0,i]
    fit = np.polyfit(x0, y0, deg=1)
    plt.plot(x0, fit[0] * x0 + fit[1],label = "CTL",color = "blue")
    plt.scatter(x0,y0,color = "blue")
    x1 = age[y_vip==1]
    y1 = U_vip[y_vip==1,i]
    fit1 = np.polyfit(x1, y1, deg=1)
    plt.plot(x1, fit1[0] * x1 + fit1[1],label = "SCZ",color = "r")
    plt.scatter(x1,y1,color = "r")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("age")
    plt.ylabel("Score on component %r"%str(i+1))
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))




#Correct effect of age
##############################################################################
# function to correct scores for age using controls
U_patients_corr = np.zeros(U_vip_scz.shape)
for i in range(10):
    regr = linear_model.LinearRegression()
    #regr.fit(age[y_vip==1].reshape(-1, 1),U_vip_scz[:,i])
    regr.fit(age[y_vip==0].reshape(-1, 1),U_vip_con[:,i])
    #regr.fit(age.reshape(-1, 1),U_vip[:,i])
    score_pred = regr.predict(age[y_vip==1].reshape(-1, 1))
    scores_corrected = U_vip_scz[:,i] - score_pred
    U_patients_corr[:,i] =  scores_corrected

##############################################################################
U_vip_scz = U_patients_corr


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/boxplots"
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos
    df["score"] = U_vip[:,i]
    df["dx"] = y_vip
    T,pvalue = scipy.stats.ttest_ind(U_vip[y_vip==0,i],U_vip[y_vip==1,i])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="dx", y="score", hue="dx", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%i)
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/panss_neg"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(panss_neg[y_vip==1])==False)]
    y = panss_neg[y_vip==1][np.array(np.isnan(panss_neg[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],panss_neg[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS NEG")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/panss_pos"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(panss_pos[y_vip==1])==False)]
    y = panss_pos[y_vip==1][np.array(np.isnan(panss_pos[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],panss_pos[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS POS")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/panss_galp"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(panss_galp[y_vip==1])==False)]
    y = panss_galp[y_vip==1][np.array(np.isnan(panss_galp[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],panss_galp[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS_GALPSYCHOPAT ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/panss_comp"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(panss_comp[y_vip==1])==False)]
    y = panss_comp[y_vip==1][np.array(np.isnan(panss_comp[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],panss_comp[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("PANSS_COMPOSITE ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/fast"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(fast[y_vip==1])==False)]
    y = fast[y_vip==1][np.array(np.isnan(fast[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],fast[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("FAST_TOT ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/cdss"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(cdss[y_vip==1])==False)]
    y = cdss[y_vip==1][np.array(np.isnan(cdss[y_vip==1])==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],cdss[y_vip==1],'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("CDSS  - Score total ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/dose"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(dose)==False)]
    y = dose[np.array(np.isnan(dose)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],dose,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Ongoing dose of treatment ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/projection_vip/hallu_panssp3"
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(hallu_panssP3_scz)==False)]
    y = hallu_panssP3_scz[np.array(np.isnan(hallu_panssP3_scz)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip[y_vip==1,i],hallu_panssP3_scz,'o')
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("hallu panss p3 ")
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))


#Correlatiojn with symotms

df = pd.DataFrame()
clinic_score = pop_vip["CVLT_RILT_RC_tot"].values[y_vip==1]
for i in range(10):
    plt.figure()
    x = U_vip_scz[:,i][np.array(np.isnan(clinic_score)==False)]
    y = clinic_score[np.array(np.isnan(clinic_score)==False)]
    Coef,pvalue = scipy.stats.pearsonr(x,y)
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    plt.plot(U_vip_scz[:,i],clinic_score,'o',label = "SCZ")
    plt.tight_layout()
    plt.xlabel("Score on component %r"%i)
    plt.ylabel("Total score for WAIS Picture Completion ")
    plt.legend()
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


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

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/WAIS_COMPL_IM_STD"
clinic_score = pop_vip["WAIS_COMPL_IM_STD"]
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
    plt.ylabel("Standard score for WAIS Picture Completion")
    plt.legend()
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/projection_vip/WAIS_MC_STD"
clinic_score = pop_vip["WAIS_MC_STD"]
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
    plt.ylabel("Standard score for WAIS Digit Span")
    plt.legend()
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%r"%(str(i+1))))

clinic_score = pop_vip["WAIS_MC_STD"]
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
    plt.ylabel("clinic_score")
    plt.legend()
    plt.title(("Coef of correlation : %s and pvalue = %r"%(np.around(Coef,decimals=3),pvalue)))


plt.plot(panss_neg[y_vip==1],clinic_score[y_vip==1],'o')
plt.xlabel("panss_neg")

plt.plot(panss_pos[y_vip==1],clinic_score[y_vip==1],'o')
plt.xlabel("panss_pos")


################################################################################
panss_neg_scz = panss_neg[y_vip==1]
panss_pos_scz = panss_pos[y_vip==1]

for i in range(10):
    plt.figure()
    plt.plot(U_vip_scz[np.array(panss_neg_scz<30),1],U_vip_scz[np.array(panss_neg_scz<30),i],'o')
    plt.plot(U_vip_scz[np.array(panss_neg_scz>25),1],U_vip_scz[np.array(panss_neg_scz>25),i],'o')

for i in range(10):
    plt.figure()
    plt.plot(U_vip_scz[np.array(panss_pos_scz<20),1],U_vip_scz[np.array(panss_pos_scz<20),i],'o')
    plt.plot(U_vip_scz[np.array(panss_pos_scz>20),1],U_vip_scz[np.array(panss_pos_scz>20),i],'o')

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




###############################################################################
#X_vip[:,1] = 1
#panss_pos_scz = panss_pos[y_vip==1][np.array(np.isnan(panss_pos[y_vip==1])==False)]
#X_scz_pos = X_vip[y_vip==1,1:][np.array(np.isnan(panss_pos[y_vip==1])==False)]
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_panss/panss_pos.npy",panss_pos_scz)
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_panss/X_scz_panss_pos.npy",X_scz_pos)
#
#
#panss_neg_scz = panss_neg[y_vip==1][np.array(np.isnan(panss_neg[y_vip==1])==False)]
#X_scz_neg = X_vip[y_vip==1,1:][np.array(np.isnan(panss_neg[y_vip==1])==False)]
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_panss/panss_neg.npy",panss_neg_scz)
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/data_panss/X_scz_panss_neg.npy",X_scz_neg)
#
#lr = linear_model.LinearRegression()
#pred = sklearn.cross_validation.cross_val_predict(lr,X_scz_pos , panss_pos_scz, cv=5)
#slope, intercept, r_value, p_value, std_err = stats.linregress(panss_pos_scz, pred)
#
#
#lr = linear_model.LinearRegression()
#pred = sklearn.cross_validation.cross_val_predict(lr,X_scz_neg , panss_neg_scz, cv=5)
#slope, intercept, r_value, p_value, std_err = stats.linregress(panss_neg_scz, pred)
#
#
#lr = linear_model.LinearRegression()
#pred = sklearn.cross_validation.cross_val_predict(lr,X_scz_neg , panss_neg_scz, cv=5)
#slope, intercept, r_value, p_value, std_err = stats.linregress(panss_neg_scz, pred)
#
#plt.plot(panss_neg_scz, pred, 'o', label='original data')
#plt.plot(panss_neg_scz, intercept + slope*panss_neg_scz, 'r', label='fitted line')
#plt.xlabel("MAASC score")
#plt.ylabel("Predicted score using MRI-based features")
#plt.legend()
#plt.show()
###############################################################################
