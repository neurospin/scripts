#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:02:41 2018

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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_COBRE_assessmentData_4495.csv"
dict_cobre = pd.read_excel("/neurospin/abide/schizConnect/data/december_2017_clinical_score/COBRE_Data_Dictionary.xlsx")


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_cobre_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/labels_cluster.npy")
labels_cluster = labels_cluster[site==1]

#Demogrpahic symptoms
################################################################################

#sum(sex[labels_cluster==0]==0)
#sum(sex[labels_cluster==0]==1)
#
#sum(sex[labels_cluster==1]==0)
#sum(sex[labels_cluster==1]==1)
#
#age[labels_cluster==0].mean()
#age[labels_cluster==1].mean()
#age[labels_cluster==0].std()
#age[labels_cluster==1].std()
#scipy.stats.f_oneway(age[labels_cluster==0],age[labels_cluster==1])
#################################################################################

df_scores = pd.DataFrame()
df_scores["subjectid"] = pop.subjectid
for score in clinic.question_id.unique():
    df_scores[score] = np.nan

for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    for key in clinic.question_id.unique():
        if curr[curr.question_id == key].empty == False:
            df_scores.loc[df_scores["subjectid"]== s,key] = curr[curr.question_id == key].question_value.values[0]

df_scores["length_disease"] = age - df_scores["CODEM_16"].astype(np.float).values
df_scores[df_scores["CODEM_19"] == "unknown"] = np.nan
df_scores[df_scores["CODEM_19"] == '9999'] = np.nan

################################################################################

df_stats = pd.DataFrame(columns=["def","T","p","mean Cluster 1","mean Cluster 2"])
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre/clusters_clinics_p_values.csv"
key_of_interest= list()

for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        defi = dict_cobre[dict_cobre["Question ID"] == key]["Question Label"].values
        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        df_stats.loc[df_stats.clinical_scores==key,"def"] = defi
        T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                 df[df["labels"]==1][key])
        if p<0.05:
            print(key)
            print(p)
            key_of_interest.append(key)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = round(T,3)
        df_stats.loc[df_stats.clinical_scores==key,"p"] = round(p,4)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 1"] = round(df[df["labels"]==0][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 2"] = round(df[df["labels"]==1][key].mean(),3)

    except:
        print("issue")
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(output)

################################################################################

df_scores["PANSS_POS"] = df_scores["FIPAN_1"].astype(np.float).values+df_scores["FIPAN_2"].astype(np.float).values+\
df_scores["FIPAN_3"].astype(np.float).values+df_scores["FIPAN_4"].astype(np.float).values+\
df_scores["FIPAN_5"].astype(np.float).values+df_scores["FIPAN_6"].astype(np.float).values+\
df_scores["FIPAN_7"].astype(np.float).values


df_scores["PANSS_NEG"] = df_scores["FIPAN_8"].astype(np.float).values+df_scores["FIPAN_9"].astype(np.float).values+\
df_scores["FIPAN_10"].astype(np.float).values+df_scores["FIPAN_11"].astype(np.float).values+\
df_scores["FIPAN_12"].astype(np.float).values+df_scores["FIPAN_13"].astype(np.float).values+\
df_scores["FIPAN_14"].astype(np.float).values

df_scores["PANSS_DES"] = df_scores["FIPAN_15"].astype(np.float).values+df_scores["FIPAN_16"].astype(np.float).values+\
df_scores["FIPAN_17"].astype(np.float).values+df_scores["FIPAN_18"].astype(np.float).values+\
df_scores["FIPAN_19"].astype(np.float).values+df_scores["FIPAN_20"].astype(np.float).values+\
df_scores["FIPAN_21"].astype(np.float).values+df_scores["FIPAN_22"].astype(np.float).values+\
df_scores["FIPAN_23"].astype(np.float).values+df_scores["FIPAN_24"].astype(np.float).values+\
df_scores["FIPAN_25"].astype(np.float).values+df_scores["FIPAN_26"].astype(np.float).values+\
df_scores["FIPAN_27"].astype(np.float).values+df_scores["FIPAN_28"].astype(np.float).values+\
df_scores["FIPAN_29"].astype(np.float).values+df_scores["FIPAN_20"].astype(np.float).values

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre/anova"



###############################################################################


df_scores[key]
for key in key_of_interest:
    plt.figure()
    df = pd.DataFrame()
    score = df_scores[key].astype(np.float).values
    df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
    LABELS_DICT = {0: "cluster 1", 1: "cluster 2"}
    df["labels_name"]  = df["labels"].map(LABELS_DICT)
    df[key] =  score[np.array(np.isnan(score)==False)]
    T,p = scipy.stats.f_oneway(df[df["labels"]==0][key],\
                         df[df["labels"]==1][key])
    ax = sns.violinplot(x="labels_name", y=key, data=df,order=["cluster 1","cluster 2"])
    plt.title("ANOVA: t = %s, and  p= %s"%(T,p))
    plt.savefig(os.path.join(output,"%s.png"%key))


###############################################################################
###############################################################################
#Save table with scores
################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import six

df = df_stats[df_stats["T"].isnull()==False]
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre/all_anova_cobre_results.png"
render_mpl_table(df, header_columns=0, col_width=2.0,output=output)

df = df_stats[df_stats["T"].isnull()==False]
df = df_stats[df_stats["p"]<0.049]
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/cobre/significant_anova_cobre_results.png"
render_mpl_table(df, header_columns=0, col_width=2.0,output=output)





def render_mpl_table(data,output, col_width=30.0, row_height=0.325, font_size=12,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center',loc='upper left')

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    plt.tight_layout()
    plt.savefig(output)
    return ax

