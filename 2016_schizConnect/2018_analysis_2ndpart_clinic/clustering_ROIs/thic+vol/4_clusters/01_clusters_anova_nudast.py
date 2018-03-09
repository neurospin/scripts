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

INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"

y_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/y.npy")

pop= pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/population.csv")

site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

pop= pop[pop["site_num"]==3]
age = pop["age"].values
sex = pop["sex_num"].values

labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/thick+vol/4_clusters/labels_cluster.npy")
labels_cluster = labels_cluster[site==3]


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

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/thick+vol/4_clusters/nudast/"
key_of_interest= list()

df_stats = pd.DataFrame(columns=["T","p","mean Cluster 1","mean Cluster 2","mean Cluster 3","mean Cluster 4"])
df_stats.insert(0,"clinical_scores",df_scores.keys())
for key in df_scores.keys():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
        T,p = scipy.stats.f_oneway(df[df["labels"]=='SCZ Cluster 1'][key],\
                 df[df["labels"]=='SCZ Cluster 2'][key],\
 df[df["labels"]=='SCZ Cluster 3'][key],\
                    df[df["labels"]=='SCZ Cluster 4'][key])
        if p<0.05:
            print(key)
            print(p)
            key_of_interest.append(key)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = round(T,3)
        df_stats.loc[df_stats.clinical_scores==key,"p"] = round(p,4)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 1"] = round(df[df["labels"]=='SCZ Cluster 1'][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 2"] = round(df[df["labels"]=='SCZ Cluster 2'][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 3"] = round(df[df["labels"]=='SCZ Cluster 3'][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 4"] = round(df[df["labels"]=='SCZ Cluster 4'][key].mean(),3)
    except:
        print("issue")
        df_stats.loc[df_stats.clinical_scores==key,"T"] = np.nan
        df_stats.loc[df_stats.clinical_scores==key,"p"] = np.nan
df_stats.to_csv(os.path.join(output,"clusters_clinics_p_values.csv"))



################################################################################
key = "totalSAPS"
key = "totalSANS"
key = "sansSUbtot"
key = "sapsSUbtot"

for key in key_of_interest:
    plt.figure()
    df = pd.DataFrame()
    score = df_scores[key].astype(np.float).values
    df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
    df[key] =  score[np.array(np.isnan(score)==False)]
    T,p = scipy.stats.f_oneway(df[df["labels"]=='SCZ Cluster 1'][key],
                         df[df["labels"]=='SCZ Cluster 2'][key],\
df[df["labels"]=='SCZ Cluster 3'][key],\
                        df[df["labels"]=='SCZ Cluster 4'][key])
    ax = sns.violinplot(x="labels", y=key, data=df,order=["Controls",'SCZ Cluster 1','SCZ Cluster 2','SCZ Cluster 3','SCZ Cluster 4'])
    plt.title("ANOVA patients diff: t = %s, and  p= %s"%(T,p))
    plt.savefig(os.path.join(output,"plots","%s.png"%key))



###############################################################################
###############################################################################
#Save table with scores
import six

df = df_stats[df_stats["T"].isnull()==False]
render_mpl_table(df, header_columns=0, col_width=2.0,output=os.path.join(output,"all_anova_nudast_results.png"))

df = df_stats[df_stats["T"].isnull()==False]
df = df_stats[df_stats["p"]<0.05]
render_mpl_table(df, header_columns=0, col_width=2.0,output=os.path.join(output,"significant_anova_nudast_results.png"))





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

