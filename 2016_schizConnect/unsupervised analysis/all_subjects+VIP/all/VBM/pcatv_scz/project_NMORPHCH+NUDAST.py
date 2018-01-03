#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:54:54 2017

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
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

INPUT_CLINIC_FILENAME_NUDAST = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"
INPUT_POPULATION_NUDAST = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"

INPUT_CLINIC_FILENAME_NMorphCH = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NMorphCH_assessmentData_4495.csv"
INPUT_POPULATION_NMorphCH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"

clinic_nudast = pd.read_csv(INPUT_CLINIC_FILENAME_NUDAST )
pop_nudast = pd.read_csv(INPUT_POPULATION_NUDAST )

clinic_nmorph = pd.read_csv(INPUT_CLINIC_FILENAME_NMorphCH )
pop_nmorph = pd.read_csv(INPUT_POPULATION_NMorphCH )

pop_all = pop_nudast.append(pop_nmorph)
clinic_all = clinic_nudast.append(clinic_nmorph)


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"
U_nudast = np.load(os.path.join(WD,"U_nudast.npy"))
U_nudast_scz = np.load(os.path.join(WD,"U_nudast_scz.npy"))
U_nudast_con = np.load(os.path.join(WD,"U_nudast_con.npy"))
y_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/y.npy")
X_nudast = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NUSDAST/X.npy")
U_nmorph = np.load(os.path.join(WD,"U_nmorph.npy"))
U_nmorph_scz = np.load(os.path.join(WD,"U_nmorph_scz.npy"))
U_nmorph_con = np.load(os.path.join(WD,"U_nmorph_con.npy"))
y_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/y.npy")
X_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/X.npy")

y_all = np.concatenate((y_nudast,y_nmorph))
U_all = np.concatenate((U_nudast,U_nmorph))
U_all_scz = np.concatenate((U_nudast_scz,U_nmorph_scz))
U_all_con = np.concatenate((U_nudast_con,U_nmorph_con))

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




# Turn interactive plotting off
plt.ioff()
################################################################################
for key in clinic_all.question_id.unique():
    print("%s" %(key))
    output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_nmorph+nudast/scores/%s" %key
    if os.path.isdir(output) == False:
        os.makedirs(output)
        neurospycho = df_scores[key].astype(np.float).values[y_all==1]
        for i in range(10):
            print(i+1)
            df = pd.DataFrame()
            df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[y_all==1][np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[y_all==1][np.array(np.isnan(neurospycho)==False)]
            df["site"] = site[y_all==1][np.array(np.isnan(neurospycho)==False)]
            df["U"] = U_all_scz[:,i][np.array(np.isnan(neurospycho)==False)]
            mod = ols("U ~ neurospycho +age+sex+site",data = df).fit()
            print(mod.pvalues["neurospycho"])
            fig = plt.figure(figsize=(10,6))
            fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
            plt.figtext(0.1,-0.1,"neurospycho effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["neurospycho"],mod.pvalues["neurospycho"]))

            plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
            plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
            plt.tight_layout()
            plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
            plt.close(fig)


            # Turn interactive plotting off
################################################################################
for key in clinic_all.question_id.unique():
    neurospycho = df_scores[key].astype(np.float).values[y_all==1]
    for i in range(10):
        df = pd.DataFrame()
        df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[y_all==1][np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[y_all==1][np.array(np.isnan(neurospycho)==False)]
        df["site"] = site[y_all==1][np.array(np.isnan(neurospycho)==False)]
        df["U"] = U_all_scz[:,i][np.array(np.isnan(neurospycho)==False)]
        mod = ols("U ~ neurospycho +age+sex+site",data = df).fit()
        if (mod.pvalues["neurospycho"]<0.01) == True:
            print (key)
            print(i+1)
            print(mod.pvalues["neurospycho"])

