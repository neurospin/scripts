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
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NMorphCH_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_nmorph = np.load(os.path.join(WD,"U_nmorph.npy"))
U_nmorph_scz = np.load(os.path.join(WD,"U_nmorph_scz.npy"))
U_nmorph_con = np.load(os.path.join(WD,"U_nmorph_con.npy"))

y_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/y.npy")
X_nmorph = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/NMORPH/X.npy")


clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
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




# Turn interactive plotting off
plt.ioff()
################################################################################
for key in clinic.question_id.unique():
    print("%s" %(key))
    output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_nmorph/scores/%s" %key
    if os.path.isdir(output) == False:
        os.makedirs(output)
        neurospycho = df_scores[key].astype(np.float).values[y_nmorph==1]
        for i in range(10):
            print(i+1)
            df = pd.DataFrame()
            df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[y_nmorph==1][np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[y_nmorph==1][np.array(np.isnan(neurospycho)==False)]
            df["U"] = U_nmorph_scz[:,i][np.array(np.isnan(neurospycho)==False)]
            mod = ols("U ~ neurospycho + age + sex",data = df).fit()
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










panss_scores = np.zeros((164,30))
i=0
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for k in range(1,31):
        if curr[curr.question_id == "FIPAN_%s"%k].empty == False:
            panss_scores[i,k-1] = curr[curr.question_id == "FIPAN_%s"%k].question_value_panss_scale.values
        else:
            panss_scores[i,k-1] = np.nan

    i = i + 1

panss_pos = np.sum(panss_scores[:,:7],axis=1)
panss_neg = np.sum(panss_scores[:,7:14],axis=1)
panss_scores_scz = panss_scores[y_nmorph==1,:]
panss_pos_scz = panss_pos[y_nmorph==1,]
panss_neg_scz = panss_neg[y_nmorph==1,]


plt.plot(panss_pos_scz,panss_neg_scz,'o')
plt.xlabel("PANSS positive")
plt.ylabel("PANSS negative")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/results/projection_nmorph/panss.png")


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_nmorph/panss_pos"
for i in range(10):
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_nmorph==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["sex"] = sex[y_nmorph==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_nmorph_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_pos +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"panss_pos effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["panss_pos"],mod.pvalues["panss_pos"]))

    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.close(fig)


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_nmorph/panss_neg"
for i in range(10):
    df = pd.DataFrame()
    df["panss_neg"] = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    df["age"] = age[y_nmorph==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["sex"] = sex[y_nmorph==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["U"] = U_nmorph_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
    mod = ols("U ~ panss_neg +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"panss_neg effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["panss_neg"],mod.pvalues["panss_neg"]))

    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
          %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.close(fig)




