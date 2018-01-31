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
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"


pop = pd.read_csv(os.path.join(DATA_PATH,"pop_nudast_scz.csv"))
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
age = pop["age"].values
sex = pop["sex_num"].values

y = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")
site = site[y==1]
labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/clusters_with_controls/2_clusters_solution/labels_cluster.npy")
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


################################################################################


df_stats = pd.DataFrame(columns=["T","p","mean Cluster 1","mean Cluster 2"])
df_stats.insert(0,"clinical_scores",clinic.question_id.unique())
################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/2_clusters_solution/nudast\
clusters_clinics_p_values.csv"
key_of_interest= list()

for key in clinic.question_id.unique():
    try:
        neurospycho = df_scores[key].astype(np.float).values
        df = pd.DataFrame()
        df[key] = neurospycho[np.array(np.isnan(neurospycho)==False)]
        df["age"] = age[np.array(np.isnan(neurospycho)==False)]
        df["sex"] = sex[np.array(np.isnan(neurospycho)==False)]
        df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
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

df_scores["sansTOTAL"] = df_scores["sans1"].astype(np.float).values+df_scores["sans2"].astype(np.float).values+\
df_scores["sans3"].astype(np.float).values+df_scores["sans4"].astype(np.float).values+\
df_scores["sans5"].astype(np.float).values+df_scores["sans6"].astype(np.float).values+\
df_scores["sans7"].astype(np.float).values+df_scores["sans8"].astype(np.float).values+\
df_scores["sans9"].astype(np.float).values+df_scores["sans10"].astype(np.float).values+\
df_scores["sans11"].astype(np.float).values+df_scores["sans12"].astype(np.float).values+\
df_scores["sans13"].astype(np.float).values+df_scores["sans14"].astype(np.float).values+\
df_scores["sans15"].astype(np.float).values+df_scores["sans16"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans18"].astype(np.float).values+\
df_scores["sans19"].astype(np.float).values+df_scores["sans20"].astype(np.float).values+\
df_scores["sans21"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans23"].astype(np.float).values+df_scores["sans24"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values

df_scores["sapsTOTAL"] = df_scores["sans1"].astype(np.float).values+df_scores["saps2"].astype(np.float).values+\
df_scores["saps3"].astype(np.float).values+df_scores["saps4"].astype(np.float).values+\
df_scores["saps5"].astype(np.float).values+df_scores["saps6"].astype(np.float).values+\
df_scores["saps7"].astype(np.float).values+df_scores["saps8"].astype(np.float).values+\
df_scores["saps9"].astype(np.float).values+df_scores["saps10"].astype(np.float).values+\
df_scores["saps11"].astype(np.float).values+df_scores["saps12"].astype(np.float).values+\
df_scores["saps13"].astype(np.float).values+df_scores["saps14"].astype(np.float).values+\
df_scores["saps15"].astype(np.float).values+df_scores["saps16"].astype(np.float).values+\
df_scores["saps17"].astype(np.float).values+df_scores["saps18"].astype(np.float).values+\
df_scores["saps19"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps21"].astype(np.float).values+df_scores["saps22"].astype(np.float).values+\
df_scores["saps23"].astype(np.float).values+df_scores["saps24"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps26"].astype(np.float).values+\
df_scores["saps27"].astype(np.float).values+df_scores["saps28"].astype(np.float).values+\
df_scores["saps29"].astype(np.float).values+df_scores["saps30"].astype(np.float).values+\
df_scores["saps31"].astype(np.float).values+df_scores["saps32"].astype(np.float).values+\
df_scores["saps33"].astype(np.float).values+df_scores["saps34"].astype(np.float).values



df_scores["sansSUbtot"] = df_scores["sans8"].astype(np.float).values+df_scores["sans13"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values

df_scores["sapsSUbtot"] = df_scores["saps7"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps34"].astype(np.float).values

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/2_clusters_solution/nudast/anova"

key = "sapsSUbtot"
key = "sansSUbtot"

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
###############################################################################
#Save table with scores
import six

df = df_stats[df_stats["T"].isnull()==False]
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nudast/all_anova_nudast_results.png"
render_mpl_table(df, header_columns=0, col_width=2.0,output=output)

df = df_stats[df_stats["T"].isnull()==False]
df = df_stats[df_stats["p"]<0.049]
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering/corrected_results/correction_age_sex_site/clusters_with_controls/\
2_clusters_solution/nudast/significant_anova_nudast_results.png"
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

