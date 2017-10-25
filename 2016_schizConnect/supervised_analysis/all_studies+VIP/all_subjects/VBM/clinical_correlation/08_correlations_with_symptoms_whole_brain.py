#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:42:04 2017

@author: ad247405
"""

import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns, matplotlib.pyplot as plt
import scipy.stats
import nibabel
import sklearn
from sklearn import linear_model

INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/X.npy"

INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/y.npy"

penalty_start = 2
INPUT_POP = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv"
pop = pd.read_csv(INPUT_POP)
age = pop.age.values

X = np.load(INPUT_DATA_X)[:,penalty_start:]
y = np.load(INPUT_DATA_y).ravel()
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
#PANSS scores are available for NUDAST + VIP subjects  = 118+39 = 157 patients


age_scz = age[y==1]
age_con = age[y==0]

SAPS_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nudast.npy")
SANS_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nudast.npy")

assert SAPS_nudast.shape == SANS_nudast.shape == (118,)

X_scz = X[y==1,:]
X_con = X[y==0,:]
X_nudast = X[site==3,:]
y_nudast = y[site==3]
age_nudast = age[site==3]
age_nudast_scz = age_nudast[y_nudast==1]
X_nudast_scz = X_nudast[y_nudast==1,:]
###############################################################################



SAPS_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_vip.npy")
SANS_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_vip.npy")

#
#dose_treatment = clinic["MEDPOSO1"].values
#
#
#dose = np.array([10,10,500,10,400,25,15,3,3,4,7.5,2,40,100,7,np.NAN,50,1250,10,300,150,800,12.5,8,\
#        10,75,10,2000,np.NAN,np.NAN,np.NAN,400,1500,np.NAN,600,50,6,300,20])
#
#dose = dose[np.logical_not(np.isnan(SANS_vip))]


SAPS_vip = SAPS_vip[np.logical_not(np.isnan(SANS_vip))]
SANS_vip = SANS_vip[np.logical_not(np.isnan(SANS_vip))]
assert SAPS_vip.shape == SANS_vip.shape == (35,)

X_vip = X[site==4,:]
y_vip = y[site==4]
X_vip_scz = X_nudast[y_vip==1,:]
X_vip_scz = X_vip_scz[np.logical_not(np.isnan(SANS_vip))]
age_vip = age[site==4]
age_vip_scz = age_vip[y_vip==1]
age_vip_scz = age_vip_scz[np.logical_not(np.isnan(SANS_vip))]

###############################################################################
###############################################################################



SAPS_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nmorphch.npy")
SANS_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nmorphch.npy")


X_nmorph = X[site==2,:]
y_nmorph = y[site==2]
X_nmorph_scz = X_nmorph[y_nmorph==1,:]
X_nmorph_scz = X_nmorph_scz[np.logical_not(np.isnan(SANS_nmorph))]
age_nmorph = age[site==3]
age_nmorph_scz = age_nmorph[y_nmorph==1]
age_nmorph_scz = age_nmorph_scz[np.logical_not(np.isnan(SANS_nmorph))]


SAPS_nmorph = SAPS_nmorph[np.logical_not(np.isnan(SANS_nmorph))]
SANS_nmorph = SANS_nmorph[np.logical_not(np.isnan(SANS_nmorph))]
assert SAPS_nmorph.shape == SANS_nmorph.shape == (41,)

###############################################################################


#Weight map
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]
assert X_vip.shape[1] == X_nudast.shape[1] == X_nmorph.shape[1] == beta.shape[0]





decision_function_vip = np.dot(X_vip_scz,beta).ravel()
decision_function_nudast = np.dot(X_nudast_scz,beta).ravel()
decision_function_nmorph = np.dot(X_nmorph_scz,beta).ravel()

decision_function_all = np.dot(X,beta).ravel()
decision_function_con = np.dot(X_con,beta).ravel()
decision_function_scz = np.dot(X_scz,beta).ravel()


#ALL
####
corr,p = scipy.stats.pearsonr(decision_function_all,age)
plt.figure()
plt.plot(decision_function_con,age[y==0],'o',label = "controls")
plt.plot(decision_function_scz,age[y==1],'o',label = "patients")
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.legend(fontsize = 15,loc = "upper left")
plt.show()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/whole_brain_age")

corr,p = scipy.stats.pearsonr(decision_function_all,age)
plt.figure()
plt.plot(decision_function_all[site==1],age[site==1],'o',label = "COBRE")
plt.plot(decision_function_all[site==2],age[site==2],'o',label = "NMorphCH")
plt.plot(decision_function_all[site==3],age[site==3],'o',label = "NUSDAST")
plt.plot(decision_function_all[site==4],age[site==4],'o',label = "VIP")
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.legend(fontsize = 15,loc = "upper left")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/whole_brain_age_pers_site")



#VIP
###############################################################################

corr,p = scipy.stats.pearsonr(decision_function_nudast,SAPS_nudast)
plt.figure()
plt.plot(decision_function_vip,SAPS_vip,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SAPS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/vip_whole_brain_SAPS")

corr,p = scipy.stats.pearsonr(decision_function_vip,SANS_vip)
plt.figure()
plt.plot(decision_function_vip,SANS_vip,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/vip_whole_brain_SANS")

corr,p = scipy.stats.pearsonr(decision_function_vip,age_vip_scz)
plt.figure()
plt.plot(decision_function_vip,age_vip_scz,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/vip_whole_brain_age")


#NUDAST
###############################################################################

corr,p = scipy.stats.pearsonr(decision_function_nudast,SAPS_nudast)
plt.figure()
plt.plot(decision_function_nudast,SAPS_nudast,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SAPS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nudast_whole_brain_SAPS")

corr,p = scipy.stats.pearsonr(decision_function_nudast,SANS_nudast)
plt.figure()
plt.plot(decision_function_nudast,SANS_nudast,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nudast_whole_brain_SANS")

corr,p = scipy.stats.pearsonr(decision_function_nudast,age_nudast_scz)
plt.figure()
plt.plot(decision_function_nudast,age_nudast_scz,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nudast_whole_brain_age")


###############################################################################
#NMORPHCH
###############################################################################

corr,p = scipy.stats.pearsonr(decision_function_nmorph,SAPS_nmorph)
plt.figure()
plt.plot(decision_function_nmorph,SAPS_nmorph,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SAPS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nmorph_whole_brain_SAPS")

corr,p = scipy.stats.pearsonr(decision_function_nmorph,SANS_nmorph)
plt.figure()
plt.plot(decision_function_nmorph,SANS_nmorph,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nmorph_whole_brain_SANS")

corr,p = scipy.stats.pearsonr(decision_function_nmorph,age_nmorph_scz)
plt.figure()
plt.plot(decision_function_nmorph,age_nmorph_scz,'o')
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/nmorph_whole_brain_age")


###############################################################################


df = pd.DataFrame()
df["age"] = age
df["y"] = y
df["decision_function"] = decision_function_all


from statsmodels.formula.api import ols
# formula = 'BEC ~ EtOH'               # Simple regression
formula = 'age ~ decision_function * C(y)'  # ANCOVA formula
lm = ols(formula, df)
fit = lm.fit()
print (fit.summary())

from scipy import stats
slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(decision_function_all,y=age)

slope_scz, intercept_scz, r_value_scz, p_value_scz, std_err_scz = stats.linregress(decision_function_scz,y=age_scz)

slope_con, intercept_con, r_value_con, p_value_con, std_err_con = stats.linregress(decision_function_con,y=age_con)

#http://r-eco-evo.blogspot.fr/2011/08/comparing-two-regression-slopes-by.html

mod = sklearn.linear_model.LinearRegression()
mod.fit(X=decision_function_all.reshape(606,1),y=age.reshape(606,1))
pred = mod.predict(decision_function_all.reshape(606,1))


mod_scz = sklearn.linear_model.LinearRegression()
mod_scz.fit(X=decision_function_scz.reshape(276,1),y=age_scz.reshape(276,1))
scz_pred = mod_scz.predict(decision_function_scz.reshape(276,1))

mod_con = sklearn.linear_model.LinearRegression()
mod_con.fit(X=decision_function_con.reshape(330,1),y=age_con.reshape(330,1))
con_pred = mod_con.predict(decision_function_con.reshape(330,1))


corr,p = scipy.stats.pearsonr(decision_function_all,age)
plt.figure()
plt.plot(decision_function_con,age[y==0],'o',label = "controls",color = "b")
plt.plot(decision_function_con,con_pred,color = "b")
plt.plot(decision_function_scz,age[y==1],'o',label = "patients",color = "g")
plt.plot(decision_function_scz,scz_pred,color = "g")
plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
plt.xlabel('Decision function')
plt.ylabel('age')
plt.tight_layout()
plt.legend(fontsize = 15,loc = "upper left")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/whole_brain_corr/whole_brain_age_with_fit.png")