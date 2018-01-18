#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:53:19 2017

@author: ad247405
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_all = np.load(os.path.join(WD,"U_all.npy"))
U_all_scz = np.load(os.path.join(WD,"U_all_scz.npy"))
U_all_con = np.load(os.path.join(WD,"U_all_con.npy"))

y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
age = pop["age"].values
sex = pop["sex_num"].values
site = pop["site_num"].values
dx = pop['dx_num']

df = pd.DataFrame()
df["age"] = age
df["age2"] = age *age
df["age3"] = age *age*age
df["sex"] = sex
df["site"] = site
for i in range(1,11):
    df["U%s"%i] = U_all[:,i-1]

for i in range(1,11):
    mod = ols("U%i ~ age+sex+C(site)+age:dx"%i,data = df).fit()
    print(mod.summary())
    print(mod.pvalues["age:dx"])

for i in range(1,11):
    mod = ols("U%i ~ age+age2+ sex+C(site)+age*dx"%i,data = df).fit()
    #print(mod.summary())
    print(mod.pvalues["age:dx"])

for i in range(1,11):
    mod = ols("U%i ~ age+age2+age3+sex+C(site)+age:dx"%i,data = df).fit()
    print(mod.summary())
    print(mod.pvalues["age:dx"])


#Compare models
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
mod1 = ols("U%i ~ age+sex+C(site)+age:dx"%i,data = df).fit()
mod2 = ols("U%i ~ age+age2+ sex+C(site)+age:dx"%i,data = df).fit()
anovaResults = anova_lm(mod1, mod2)
print(anovaResults)

mod1 = ols("U%i ~ age+sex+C(site)+age:dx"%i,data = df).fit()
mod3 = ols("U%i ~ age+age2+age3+sex+C(site)+age:dx"%i,data = df).fit()
anovaResults = anova_lm(mod1, mod3)
print(anovaResults)

mod2 = ols("U%i ~ age+age2+ sex+C(site)+age:dx"%i,data = df).fit()
mod3 = ols("U%i ~ age+age2+age3+sex+C(site)+age:dx"%i,data = df).fit()
anovaResults = anova_lm(mod2, mod3)
print(anovaResults)

#Mod2 seems to be the best model


#Plot
df_con = pd.DataFrame()
df_con["Age"] = age[y_all==0]
df_con["Age2"] = (age[y_all==0])*(age[y_all==0])
df_con["sex"] = sex[y_all==0]
df_con["site"] = site[y_all==0]
for i in range(1,11):
    df_con["Score on comp U%s"%i] = U_all_con[:,i-1]

df_scz = pd.DataFrame()
df_scz["Age"] = age[y_all==1]
df_scz["Age2"] = (age[y_all==1])*(age[y_all==1])
df_scz["sex"] = sex[y_all==1]
df_scz["site"] = site[y_all==1]
for i in range(1,11):
    df_scz["Score on comp U%s"%i] = U_all_scz[:,i-1]

output = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_all+vip/age/linear"
import seaborn as sns
sns.set(color_codes=True)
for i in range(1,11):
    plt.figure()
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_con,label= "SCZ",marker='o')
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_scz,label="CTL",marker='d')
    plt.legend()
    plt.savefig(os.path.join(output,"comp%s"%i))

output = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_all+vip/age/quadratic"
sns.set(color_codes=True)
for i in range(1,11):
    plt.figure()
    sns.regplot(x="Age2", y="Score on comp U%s"%i, data=df_con,label= "SCZ",marker='o')
    sns.regplot(x="Age2", y="Score on comp U%s"%i, data=df_scz,label="CTL",marker='d')
    plt.legend()
    plt.savefig(os.path.join(output,"comp%s"%i))
