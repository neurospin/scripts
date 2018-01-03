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

INPUT_CLINIC_FILENAME_COBRE = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_COBRE_assessmentData_4495.csv"
INPUT_POPULATION_COBRE = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"

INPUT_CLINIC_FILENAME_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/sujets_series.xls"
INPUT_POPULATION_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv"

clinic_cobre = pd.read_csv(INPUT_CLINIC_FILENAME_COBRE )
pop_cobre = pd.read_csv(os.path.join(DATA_PATH,"pop_cobre_scz.csv"))

clinic_vip = pd.read_excel(INPUT_CLINIC_FILENAME_VIP )
pop_vip = pd.read_csv(INPUT_POPULATION_VIP )
pop_vip["subjectid"] = pop_vip['code_vip']
pop_vip["sex_num"] = pop_vip['sex_code']
pop_vip = pop_vip[pop_vip.dx==1]
pop_all = pop_cobre.append(pop_vip)
assert pop_all.shape == (116, 938)



age = pop_all["age"].values
sex = pop_all["sex_num"].values
SITE_MAP = {"MRN": 0, np.nan:1 }
pop_all["site_num"] = pop_all["site"].map(SITE_MAP)
site = pop_all["site_num"].values

panss = "Delusions","Conceptual_Disorganization",	"Hallucinatory_Behavior",\
	"Excitement",	"Grandiosity",	"Suspiciousness_Persecution","Hostility","Blunted_Affect",\
	"Emotional_Withdrawal","Poor_Rapport","Passive_Apathetic_Social_Withdrawal",\
	"Difficulty_in_Abstract_Thinking",	"Lack_of_Spontaneity_and_Flow_of_Conversation",\
	"Stereotyped_Thinking","Somatic_Concern","Anxiety","Guilt_Feelings",\
	"Tension", "Mannerisms_and_Posturing", "Depression","Motor_Retardation",\
 "Uncooperativeness","Unusual_Thought_Content","Disorientation","Poor_Attention",\
	"Lack_of_Judgment_and_Insight","Disturbance_of_Volition","Poor_Impulse_Control",\
	"Preoccupation","Active_Social_Avoidance"
assert len(panss) == 30
#VIP
############################
df_scores_vip = pd.DataFrame()
df_scores_vip["subjectid"] = pop_vip.subjectid
for dim in panss :
    df_scores_vip[dim] = np.nan

for i in range(7):
    df_scores_vip[panss[i]] = pop_vip["PANSS_P%s"%(i+1)]
for i in range(7,14):
    df_scores_vip[panss[i]] = pop_vip["PANSS_N%s"%(i-6)]
for i in range(14,30):
    df_scores_vip[panss[i]] = pop_vip["PANSS_G%s"%(i-13)]

#############
#COBRE
###############
PANSS_MAP = {"Absent": 1, "Minimal": 2, "Mild": 3, "Moderate": 4, "Moderate severe": 5, "Severe": 6, "Extreme": 7,\
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
clinic_cobre["question_value"] = clinic_cobre["question_value"].map(PANSS_MAP)


panss  = 'FIPAN_1', 'FIPAN_10', 'FIPAN_11', 'FIPAN_12',\
       'FIPAN_13', 'FIPAN_14', 'FIPAN_15', 'FIPAN_16', 'FIPAN_17',\
       'FIPAN_18', 'FIPAN_19', 'FIPAN_2', 'FIPAN_20', 'FIPAN_21',\
       'FIPAN_22', 'FIPAN_23', 'FIPAN_24', 'FIPAN_25', 'FIPAN_26',\
       'FIPAN_27', 'FIPAN_28', 'FIPAN_29', 'FIPAN_3', 'FIPAN_30',\
       'FIPAN_4', 'FIPAN_5', 'FIPAN_6', 'FIPAN_7', 'FIPAN_8', 'FIPAN_9',

df_scores_cobre = pd.DataFrame()
df_scores_cobre["subjectid"] = pop_cobre.subjectid
for score in panss:
    df_scores_cobre[score] = np.nan

for s in pop_all.subjectid:
    curr = clinic_cobre[clinic_cobre.subjectid ==s]
    for key in panss:
        if curr[curr.question_id == key].empty == False:
            df_scores_cobre.loc[df_scores_cobre["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]


PANSS_MAP = { "FIPAN_1":"Delusions",'FIPAN_2':"Conceptual_Disorganization",\
             'FIPAN_3':"Hallucinatory_Behavior",'FIPAN_4':"Excitement",\
             'FIPAN_5':"Grandiosity",'FIPAN_6':"Suspiciousness_Persecution",
             'FIPAN_7':"Hostility",'FIPAN_8':"Blunted_Affect",\
	'FIPAN_9':"Emotional_Withdrawal",'FIPAN_10':"Poor_Rapport",\
 'FIPAN_11':"Passive_Apathetic_Social_Withdrawal",'FIPAN_12':"Difficulty_in_Abstract_Thinking",\
'FIPAN_13':"Lack_of_Spontaneity_and_Flow_of_Conversation",'FIPAN_14':"Stereotyped_Thinking",\
'FIPAN_15':"Somatic_Concern",'FIPAN_16':"Anxiety",'FIPAN_17':"Guilt_Feelings",\
	'FIPAN_18':"Tension",'FIPAN_19': "Mannerisms_and_Posturing",'FIPAN_20': "Depression",\
 'FIPAN_21':"Motor_Retardation",'FIPAN_22':"Uncooperativeness",\
 'FIPAN_23':"Unusual_Thought_Content",'FIPAN_24':"Disorientation",'FIPAN_25':"Poor_Attention",\
	'FIPAN_26':"Lack_of_Judgment_and_Insight",'FIPAN_27':"Disturbance_of_Volition",
 'FIPAN_28':"Poor_Impulse_Control",'FIPAN_29':"Preoccupation",'FIPAN_30':"Active_Social_Avoidance"}

df_scores_cobre = df_scores_cobre.rename(index=str,columns =PANSS_MAP)


df_scores_all = df_scores_cobre.append(df_scores_vip)


df_scores_all["sum_pos"] = df_scores_all["Delusions"] + df_scores_all["Conceptual_Disorganization"]+\
	df_scores_all["Hallucinatory_Behavior"] + df_scores_all["Excitement"]+\
    df_scores_all["Grandiosity"] + df_scores_all["Suspiciousness_Persecution"]+\
    df_scores_all["Hostility"]


df_scores_all["sum_neg"] = df_scores_all["Blunted_Affect"] + df_scores_all["Emotional_Withdrawal"]+\
	df_scores_all["Poor_Rapport"] + df_scores_all["Passive_Apathetic_Social_Withdrawal"]+\
    df_scores_all["Difficulty_in_Abstract_Thinking"] + df_scores_all["Lack_of_Spontaneity_and_Flow_of_Conversation"]+\
    df_scores_all["Stereotyped_Thinking"]


df_scores_all["sum_gen"] = df_scores_all["Somatic_Concern"] + df_scores_all["Anxiety"]+\
	df_scores_all["Guilt_Feelings"] + df_scores_all["Tension"]+\
    df_scores_all["Mannerisms_and_Posturing"] + df_scores_all["Depression"]+\
    df_scores_all["Motor_Retardation"]+ df_scores_all[ "Uncooperativeness"]+\
	df_scores_all["Unusual_Thought_Content"] + df_scores_all["Disorientation"]+\
    df_scores_all["Poor_Attention"] + df_scores_all["Lack_of_Judgment_and_Insight"]+\
    df_scores_all["Disturbance_of_Volition"] + df_scores_all["Poor_Impulse_Control"]+\
    df_scores_all["Preoccupation"]+df_scores_all["Active_Social_Avoidance"]

df_scores_all["composite_score"] = df_scores_all["sum_pos"] - df_scores_all["sum_neg"]

df_scores_all.to_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/data/data_panss/cobre+vip_panss.csv")

panss = "composite_score","sum_gen","sum_pos","sum_neg","Delusions","Conceptual_Disorganization",	"Hallucinatory_Behavior",\
	"Excitement",	"Grandiosity",	"Suspiciousness_Persecution","Hostility","Blunted_Affect",\
	"Emotional_Withdrawal","Poor_Rapport","Passive_Apathetic_Social_Withdrawal",\
	"Difficulty_in_Abstract_Thinking",	"Lack_of_Spontaneity_and_Flow_of_Conversation",\
	"Stereotyped_Thinking","Somatic_Concern","Anxiety","Guilt_Feelings",\
	"Tension", "Mannerisms_and_Posturing", "Depression","Motor_Retardation",\
 "Uncooperativeness","Unusual_Thought_Content","Disorientation","Poor_Attention",\
	"Lack_of_Judgment_and_Insight","Disturbance_of_Volition","Poor_Impulse_Control",\
	"Preoccupation","Active_Social_Avoidance"
assert len(panss) == 34




clusters = 'cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',\
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',\
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',\
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',\
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus'

#pvalues
##############################################################################
df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",panss)
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/vip+cobre_clusters_clinics_p_values.csv"
for key in panss:
    try:
        neurospycho = df_scores_all[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop_all[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)


#Tvalues
##############################################################################
df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",panss)
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/supervised_clusters_results/vip+cobre_clusters_clinics_T_values.csv"
for key in panss:
    try:
        neurospycho = df_scores_all[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop_all[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
            print(mod.tvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.tvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)