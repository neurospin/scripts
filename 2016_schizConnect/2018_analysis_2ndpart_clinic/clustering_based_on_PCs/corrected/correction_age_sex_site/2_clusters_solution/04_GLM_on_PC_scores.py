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



DATA_PATH = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data"
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"


pop = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site_scz = site[y==1]
site_con = site[y==0]

labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/correction_age_sex_site/\
2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster

U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
U_all = scipy.stats.zscore(U_all)
U_all_scz = U_all[y_all==1,:]
U_all_con = U_all[y_all==0,:]


df = pd.DataFrame()
df["age"] = age[y==1]
df["age2"] = (age *age)[y==1]
df["age3"] = (age *age*age)[y==1]
df["sex"] = sex[y==1]
df["site"] = site[y==1]
df["label"] = labels_cluster


for i in range(1,11):
    df["U%s"%i] = U_all_scz[:,i-1]

for i in range(1,11):
    mod = ols("U%i ~ age+sex+C(site)+age*label"%i,data = df).fit()
    #print(mod.summary())
    print(mod.pvalues["age:label"])

for i in range(1,11):
    mod = ols("U%i ~ age+age2+ sex+C(site)+age*label"%i,data = df).fit()
    #print(mod.summary())
    print(mod.pvalues["age:label"])


for i in range(1,11):
    mod = ols("U%i ~ age+age2+age3+sex+C(site)+age*label"%i,data = df).fit()
    #p0rint(mod.summary())
    print(mod.pvalues["age:label"])
   print(mod.pvalues["age:label"])




#Plot
df_clust1 = pd.DataFrame()
df_clust1["Age"] = age[y==1][labels_cluster ==0]
df_clust1["Age2"] = (age[y==1][labels_cluster ==0])*(age[y==1][labels_cluster ==0])
df_clust1["sex"] = sex[y==1][labels_cluster ==0]
df_clust1["site"] = site[y==1][labels_cluster ==0]
for i in range(1,11):
    df_clust1["Score on comp U%s"%i] = U_all_scz[labels_cluster ==0,i-1]

#Plot
df_clust2 = pd.DataFrame()
df_clust2["Age"] = age[y==1][labels_cluster ==1]
df_clust2["Age2"] = (age[y==1][labels_cluster ==1])*(age[y==1][labels_cluster ==1])
df_clust2["sex"] = sex[y==1][labels_cluster ==1]
df_clust2["site"] = site[y==1][labels_cluster ==1]
for i in range(1,11):
    df_clust2["Score on comp U%s"%i] = U_all_scz[labels_cluster ==1,i-1]


output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/\
2_clusters_solution/age_trajectories/linear"
import seaborn as sns
sns.set(color_codes=True)
for i in range(1,11):
    plt.figure()
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_clust1,label= "Cluster 1",marker='o')
    sns.regplot(x="Age", y="Score on comp U%s"%i, data=df_clust2,label="Cluster 2",marker='d')
    plt.legend()
    plt.savefig(os.path.join(output,"comp%s"%i))

output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/\
2_clusters_solution/age_trajectories/quadratic"
sns.set(color_codes=True)
for i in range(1,11):
    plt.figure()
    sns.regplot(x="Age2", y="Score on comp U%s"%i, data=df_clust1,label= "Cluster 1",marker='o')
    sns.regplot(x="Age2", y="Score on comp U%s"%i, data=df_clust2,label="Cluster 2",marker='d')
    plt.legend()
    plt.savefig(os.path.join(output,"comp%s"%i))
