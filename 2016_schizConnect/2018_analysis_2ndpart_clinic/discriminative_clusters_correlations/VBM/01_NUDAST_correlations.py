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

pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
#pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast.csv"))

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values


df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]

df_scores["sansSUbtot"] = df_scores["sans8"].astype(np.float).values+df_scores["sans13"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values


df_scores["sapsSUbtot"] = df_scores["saps7"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps34"].astype(np.float).values

clusters = 'cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',\
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',\
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',\
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',\
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus','cluster11_predictive_signature'

#
#df_stats = pd.DataFrame(columns=clusters)
#df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
#################################################################################
#output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/clusters_clinics_p_values.csv"
#for key in df_scores.keys():
#    try:
#        neurospycho = df_scores[key].astype(np.float).values
#        for clust in clusters:
#            print(clust)
#            df = pd.DataFrame()
#            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
#            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
#            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
#            df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
#            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
#            print(mod.pvalues[key])
#            mod.summary2()
#            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]
#
#    except:
#            print("issue")
#            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan
#
#
#df_stats.to_csv(output)
#
#################################################################################
#df_stats = pd.DataFrame(columns=clusters)
#df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
#output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
#results/supervised_clusters_results/clusters_clinics_p_values_by_clust.csv"
#
#clusters = ['cluster11_predictive_signature']
#NP_scores = ["vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc","fpiscalc",'matrxsca',"trailb","wcstpsve"]
#for key in ["sansSUbtot","sapsSUbtot","vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc","fpiscalc",'matrxsca',"trailb","wcstpsve"]:
#    try:
#        neurospycho = df_scores[key].astype(np.float).values
#        print(key)
#        for clust in clusters:
#            print(clust)
#            df = pd.DataFrame()
#            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
#            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
#            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
#            df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
#            mod = ols("%s ~ %s +age+sex"%(key,clust),data = df).fit()
#            print(mod.pvalues[clust])
#            print(mod.summary2())
#            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]
#
#    except:
#            print("issue")
#            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan
key = "sansSUbtot"
key = "sapsSUbtot"
key = "vocabsca"
key = 'dstscalc'
key = 'sstscalc'
key = "lnsscalc"
key = "d4prime"
key = "lmiscalc"
key = "fpiscalc"
key = 'matrxsca'
key = "trailb"
key = "wcstpsve"

clust = 'cluster11_predictive_signature'
df = pd.DataFrame()
y = pop["dx_num"].values
neurospycho = df_scores[key].astype(np.float).values
df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
df["age"] = age[np.array(np.isnan(neurospycho)==False)]
df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
y = y[np.array(np.isnan(neurospycho)==False)]
df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
mod = ols("%s ~age+sex"%(clust),data = df).fit()
res = mod.resid


plt.plot(res,df[key].values,'o',marker='o',label = "Controls",\
         color="royalblue",markersize=6)
m, b = np.polyfit(res,df[key].values, 1)
plt.plot(res, m*res + b, '-',color="black")
plt.xlabel("Neuroanatomical predictive signature score")
plt.ylabel(" Positive Symptoms score (SAPS)")
R,p = scipy.stats.pearsonr(res,df[key].values)
plt.title("R = %s and p=%s" %(R,p/2))
plt.savefig("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/figure_correlation_paper")