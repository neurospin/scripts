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
    output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_cobre/scores/%s" %key
    if os.path.isdir(output) == False:
        os.makedirs(output)
        neurospycho = df_scores[key].astype(np.float).values[y_cobre==1]
        for i in range(10):
            print(i+1)
            df = pd.DataFrame()
            df["neurospycho"] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[y_cobre==1][np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[y_cobre==1][np.array(np.isnan(neurospycho)==False)]
            df["U"] = U_cobre_scz[:,i][np.array(np.isnan(neurospycho)==False)]
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













PANSS_MAP = {"Absent": 1, "Minimal": 2, "Mild": 3, "Moderate": 4, "Moderate severe": 5, "Severe": 6, "Extreme": 7,\
             "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
clinic["question_value"] = clinic["question_value"].map(PANSS_MAP)


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
panss_scores_scz = panss_scores[y_cobre==1,:]
panss_pos_scz = panss_pos[y_cobre==1,]
panss_neg_scz = panss_neg[y_cobre==1,]


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
    df["sex"] = sex[y_cobre==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_cobre_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
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
all_subjects/results/pcatv_scz/results/projection_cobre/panss_neg"
for i in range(10):
    df = pd.DataFrame()
    df["panss_neg"] = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    df["age"] = age[y_cobre==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["sex"] = sex[y_cobre==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["U"] = U_cobre_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
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



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
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
pop = pop[pop["dx_num"]==1]
age = pop["age"].values
sex = pop["sex_num"].values

pop["PC1_whole_brain"] = U_cobre_scz[:, 0]
pop["PC2_caudate_putamen"] = U_cobre_scz[:,1]
pop["PC3_hippocampus_amygdala"] = U_cobre_scz[:, 2]
pop["PC4_temporal_gyrus"] = U_cobre_scz[:, 3]
pop["PC5_frontal_orbital_cortex"] = U_cobre_scz[:, 4]
pop["PC6_thalamus"] = U_cobre_scz[:, 5]
pop["PC7_left_thalamus"] = U_cobre_scz[:, 6]
pop["PC8_occipital"] = U_cobre_scz[:, 7]
pop["PC9_superior_frontal_gyrus"] = U_cobre_scz[:, 8]
pop["PC10_heschl_gyrus"] = U_cobre_scz[:, 9]


df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]
    print(s)


clusters = 'PC1_whole_brain', 'PC2_caudate_putamen',\
       'PC3_hippocampus_amygdala', 'PC4_temporal_gyrus',\
       'PC5_frontal_orbital_cortex', 'PC6_thalamus',\
       'PC7_left_thalamus', 'PC8_occipital',\
       'PC9_superior_frontal_gyrus', 'PC10_heschl_gyrus'


df_stats = pd.DataFrame(columns=clusters)
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_cobre/PC_clinics_p_values.csv"
for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        for clust in clusters:
            print(clust)
            df = pd.DataFrame()
            df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
            df[clust] = pop[clust][np.array(np.isnan(neurospycho)==False)].values
            mod = ols("%s ~ %s +age+sex"%(clust,key),data = df).fit()
            print(mod.pvalues[key])
            df_stats.loc[df_stats.clinical_scores==key,clust] = mod.pvalues[key]

    except:
            print("issue")
            df_stats.loc[df_stats.clinical_scores==key,clust] = np.nan


df_stats.to_csv(output)
