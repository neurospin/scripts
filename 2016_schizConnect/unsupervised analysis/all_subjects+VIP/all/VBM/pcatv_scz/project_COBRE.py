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

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_COBRE_assessmentData_4495.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_cobre = np.load(os.path.join(WD,"U_cobre.npy"))
U_cobre_scz = np.load(os.path.join(WD,"U_cobre_scz.npy"))
U_cobre_con = np.load(os.path.join(WD,"U_cobre_con.npy"))

y_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/y.npy")
X_cobre = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/COBRE/X.npy")


clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
age = pop["age"].values


PANSS_MAP = {"Absent": 1, "Minimal": 2, "Mild": 3, "Moderate": 4, "Moderate severe": 5, "Severe": 6, "Extreme": 7,\
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
clinic["question_value_panss_scale"] = clinic["question_value"].map(PANSS_MAP)


panss_scores = np.zeros((164,30))
i=0
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for k in range(1,31):
        if curr[curr.question_id == "FIPAN_%s"%k].empty == False:
            panss_scores[i,k-1] = curr[curr.question_id == "FIPAN_%s"%k].question_value_panss_scale.values
        else:
            panss_scores[i,k-1] = np.nan
            if(y_cobre[i]==1.0):
    i = i + 1

panss_pos = np.sum(panss_scores[:,:7],axis=1)
panss_neg = np.sum(panss_scores[:,7:14],axis=1)
panss_scores_scz = panss_scores[y_cobre==1,:]
panss_pos_scz = panss_pos[y_cobre==1,]
panss_neg_scz = panss_neg[y_cobre==1,]


CNP = np.zeros((164,105))
i=0
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for k in range(1,105):
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
VBM/all_subjects/results/pcatv_scz/results/projection_cobre/panss.png")


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_cobre/panss_pos"
for i in range(10):
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_cobre==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_cobre_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_pos +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"PANSS POSITIVE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["panss_pos"],mod.pvalues["panss_pos"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_cobre/panss_neg"
for i in range(10):
    df = pd.DataFrame()
    df["panss_neg"] = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    df["age"] = age[y_cobre==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["U"] = U_cobre_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
    mod = ols("U ~ panss_neg +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"PANSS NEGATIVE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["panss_neg"],mod.pvalues["panss_neg"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))




################################################################################
for k in range(90,105):
    print("CNP %s" %(k))
    output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_cobre/CNP/CNP_%s" %k
    os.makedirs(output)
    neurospycho = CNP_scz [:,k]
    for i in range(10):
        print(i+1)
        df = pd.DataFrame()
        df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[y_cobre==1][np.array(np.isnan(neurospycho)==False)]
        df["U"] = U_cobre_scz[:,i][np.array(np.isnan(neurospycho)==False)]
        mod = ols("U ~ neurospycho +age",data = df).fit()
        print(mod.pvalues["neurospycho"])
        fig = plt.figure(figsize=(10,6))
        fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
        plt.figtext(0.6, .3,"neurospycho effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["neurospycho"],mod.pvalues["neurospycho"]))

        plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
        plt.tight_layout()
        plt.savefig(os.path.join(output,"comp%s"%(i+1)))

