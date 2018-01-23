#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:09:36 2017

@author: ad247405
"""

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
INPUT_CLINIC_FILENAME_NUDAST = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/\
schizconnect_NUSDAST_assessmentData_4495.csv"
INPUT_CLINIC_FILENAME_NMORPH = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/\
schizconnect_NMorphCH_assessmentData_4495.csv"

pop_nudast = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
pop_nmorph = pd.read_csv(os.path.join(DATA_PATH,"pop_nmorph_scz.csv"))

clinic_nudast = pd.read_csv(INPUT_CLINIC_FILENAME_NUDAST)
clinic_nmorph = pd.read_csv(INPUT_CLINIC_FILENAME_NMORPH)

pop_all = pop_nudast.append(pop_nmorph)
clinic_all = clinic_nudast.append(clinic_nmorph)

age = pop_all["age"].values
sex = pop_all["sex_num"].values
SITE_MAP = {"NU": 0, "WUSTL":1 }
pop_all["site_num"] = pop_all["site"].map(SITE_MAP)
site = pop_all["site_num"].values

df_scores = pd.DataFrame()
df_scores["subjectid"] = pop_all.subjectid
for score in clinic_all.question_id.unique():
    df_scores[score] = np.nan

for s in pop_all.subjectid:
    curr = clinic_all[clinic_all.subjectid ==s]
    for key in clinic_all.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]

#Save PANSS info
##############################################################################
df_scores["saps_tot"] = 0
for i in range(1,35):
    df_scores["saps_tot"] = df_scores["saps_tot"] + df_scores["saps%s"%i].astype(float).values

df_scores["sans_tot"] = 0
for i in range(1,26):
    df_scores["sans_tot"] = df_scores["sans_tot"] + df_scores["sans%s"%i].astype(float).values

df_scores_panss = df_scores[["saps_tot","sans_tot"]]
df_scores_panss.to_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
data/data_panss/nudast+nmorph_panss.csv")
##############################################################################



clusters = 'cluster1_cingulate_gyrus', 'cluster2_right_caudate_putamen',\
       'cluster3_precentral_postcentral_gyrus', 'cluster4_frontal_pole',\
       'cluster5_temporal_pole', 'cluster6_left_hippocampus_amygdala',\
       'cluster7_left_caudate_putamen', 'cluster8_left_thalamus',\
       'cluster9_right_thalamus', 'cluster10_middle_temporal_gyrus'

df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",clinic_all.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/supervised_clusters_results/nudast+nmorph_clusters_clinics_p_values.csv"
for key in clinic_all.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df["site"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop_all[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex+site"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)
