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

#https://central.xnat.org/REST/projects/NUDataSharing
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/december_2017_clinical_score/schizconnect_NUSDAST_assessmentData_4495.csv"

y_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/y.npy")
site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
y_nudast = y_all[site==3]

pop= pd.read_csv("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/population.csv")

site = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/data/site.npy")
clinic = pd.read_csv(INPUT_CLINIC_FILENAME)

pop= pop[pop["site_num"]==3]
age = pop["age"].values
sex = pop["sex_num"].values

labels_cluster = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering_ROIs/results/thick+vol/3_clusters/labels_cluster.npy")
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


df_scores["totalSANS"] = 0
for i in (1,2,3,4,5,6,7,9,10,11,12,14,15,16,18,19,20,21,23,24):
   df_scores["totalSANS"] = df_scores["totalSANS"]  + df_scores["sans%s"%i].astype(np.float).values


df_scores["totalSAPS"] = 0
for i in (1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,26,27,28,29,30,31,32,33):
   df_scores["totalSAPS"] = df_scores["totalSAPS"]  + df_scores["saps%s"%i].astype(np.float).values


df_scores["sansSUbtot"] = df_scores["sans8"].astype(np.float).values+df_scores["sans13"].astype(np.float).values+\
df_scores["sans17"].astype(np.float).values+df_scores["sans22"].astype(np.float).values+\
df_scores["sans25"].astype(np.float).values


df_scores["sapsSUbtot"] = df_scores["saps7"].astype(np.float).values+df_scores["saps20"].astype(np.float).values+\
df_scores["saps25"].astype(np.float).values+df_scores["saps34"].astype(np.float).values

################################################################################
output = "/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/\
results/clustering_ROIs/results/thick+vol/3_clusters/nudast/"
key_of_interest= list()

df_stats = pd.DataFrame(columns=["T","p","mean Cluster 1","mean Cluster 2","mean Cluster 3"])
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
                    df[df["labels"]=='SCZ Cluster 3'][key])
        if p<0.05:
            print(key)
            print(p)
            key_of_interest.append(key)
        df_stats.loc[df_stats.clinical_scores==key,"T"] = round(T,3)
        df_stats.loc[df_stats.clinical_scores==key,"p"] = round(p,4)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 1"] = round(df[df["labels"]=='SCZ Cluster 1'][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 2"] = round(df[df["labels"]=='SCZ Cluster 2'][key].mean(),3)
        df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 3"] = round(df[df["labels"]=='SCZ Cluster 3'][key].mean(),3)

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
df_scores["panss_diff"] = df_scores["sansSUbtot"].astype(np.float).values-df_scores["sapsSUbtot"].astype(np.float).values
NP_scores = ["vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc","fpiscalc",'matrxsca',"trailb","wcstpsve"]


for key in key_of_interest:
    plt.figure()
    df = pd.DataFrame()
    score = df_scores[key].astype(np.float).values
    df["labels"]=labels_cluster[np.array(np.isnan(score)==False)]
    df[key] =  score[np.array(np.isnan(score)==False)]
    T,p = scipy.stats.f_oneway(df[df["labels"]=='Subcortical'][key],
                         df[df["labels"]=='Cortical'][key],\
                        df[df["labels"]=='Preserved'][key])
    ax = sns.violinplot(x="labels", y=key, data=df,order=["Controls","Subcortical","Cortical","Preserved"])
    plt.title("ANOVA patients diff: t = %s, and  p= %s"%(T,p))
    plt.savefig(os.path.join(output,"plots","%s.png"%key))


df[df["labels"]=='SCZ Cluster 1'][key].mean()
df[df["labels"]=='SCZ Cluster 1'][key].std()
df[df["labels"]=='SCZ Cluster 2'][key].mean()
df[df["labels"]=='SCZ Cluster 2'][key].std()
df[df["labels"]=='SCZ Cluster 3'][key].mean()
df[df["labels"]=='SCZ Cluster 3'][key].std()

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

###############################################################################
###############################################################################
#code to plot NP meand variables

NP_scores = ["vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc","fpiscalc",'matrxsca',"trailb","wcstpsve"]

#NP_scores = ["vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc","fpiscalc",'matrxsca']

df_stats = pd.DataFrame(columns=["mean Controls","std Controls","mean Cluster 1","mean Cluster 2","mean Cluster 3",\
                                 "std Cluster 1","std Cluster 2","std Cluster 3"])
df_stats.insert(0,"clinical_scores",NP_scores)
for key in NP_scores:
    neurospycho = df_scores[key].astype(np.float).values
    df = pd.DataFrame()
    y = y_nudast[np.array(np.isnan(neurospycho)==False)]
    score = neurospycho[np.array(np.isnan(neurospycho)==False)]
    score = ((score - score[y==0].mean(axis=0))/score[y==0].std(axis=0))
    df[key] = score
    df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
    df_stats.loc[df_stats.clinical_scores==key,"mean Controls"] = round(df[df["labels"]=='Controls'][key].mean(),3)

    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 1"] = round(df[df["labels"]=='Subcortical'][key].mean(),3)
    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 2"] = round(df[df["labels"]=='Cortical'][key].mean(),3)
    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 3"] = round(df[df["labels"]=='Preserved'][key].mean(),3)

    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 1"] = round(df[df["labels"]=='Subcortical'][key].std(),3)
    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 2"] = round(df[df["labels"]=='Cortical'][key].std(),3)
    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 3"] = round(df[df["labels"]=='Preserved'][key].std(),3)

#plt.errorbar(x,y=df_stats["mean Controls"],yerr=df_stats["std Controls"],label = "Controls",marker='o',ls='--')
#plt.errorbar(x,y=df_stats["mean Cluster 1"],yerr=df_stats["std Cluster 1"],label = "SCZ Cluster 1",marker='v',ls='--')
#plt.errorbar(x,y=df_stats["mean Cluster 2"],yerr=df_stats["std Cluster 2"],label = "SCZ Cluster 2",marker='p',ls='--')
#plt.errorbar(x,y=df_stats["mean Cluster 3"],yerr=df_stats["std Cluster 3"],label = "SCZ Cluster 3",marker='d',ls='--')
#plt.legend()
#plt.ylabel("Z-score")


plt.plot(df_stats["mean Controls"],'o',label = 'Controls', marker='o',markersize=10,ls='--',color= "darkgreen")
plt.plot(df_stats["mean Cluster 1"],'o',label = 'Subcortical', marker='v',markersize=10,ls='--',color= "darkblue")
plt.plot(df_stats["mean Cluster 2"],'o',label = 'Cortical', marker='s',markersize=10,ls='--',color= "firebrick")
plt.plot(df_stats["mean Cluster 3"],'o',label = 'Preserved', marker='d',markersize=10,ls='--',color= "goldenrod")
plt.annotate('', xy=(0.67, -0.37), xycoords='axes fraction', xytext=(1, -0.37),
            arrowprops=dict(arrowstyle="<->", color='black',lw= 3))
plt.annotate('', xy=(0.47, -0.37), xycoords='axes fraction', xytext=(0.67, -0.37),
            arrowprops=dict(arrowstyle="<->", color='black',lw= 3))
plt.annotate('', xy=(0.12, -0.37), xycoords='axes fraction', xytext=(0.47, -0.37),
            arrowprops=dict(arrowstyle="<->", color='black',lw= 3))
plt.annotate('', xy=(0, -0.37), xycoords='axes fraction', xytext=(0.12, -0.37),
            arrowprops=dict(arrowstyle="<->", color='black',lw= 3))
plt.text(7, -4.2, "Executive Functions")
plt.text(4.2, -4.2, "Episodic Memory")
plt.text(1.1, -4.2, "Working Memory")
plt.text(-0.9, -4.2, "Intelligence")

plt.legend(loc = 'upper center',ncol = 4)
plt.ylabel("Z-score")
x = np.arange(10)
plt.axis([-1,10, -2,3])
NP_scores_legend = ["WAIS vocabulary",'WMS Digit Span',"WMS Spatial Span",\
"WMS Letter Number\n Sequencing","CPT dprime", "WMS Logical Memory",\
"WMS Family Picture",'WAIS Matrix\n Reasoning',"Trails B time",'WCST perseverative\n errors']
#NP_scores_legend = ["WAIS vocab",'WMS Digit Span',"WMS Spatial Span","WMS LN","CPT dprime", "WMS LM","WMS FP",'WAIS matrix']
plt.xticks(x,NP_scores_legend,rotation=60, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output,"neuropsy_per_clusters"), bbox_inches="tight")
###############################################################################
NP_scores = ["cvl15tsc","cvlsdcrs",\
"cvlldcrs","cvllsls","cvlrccs"]

df_stats = pd.DataFrame(columns=["mean Controls","std Controls","mean Cluster 1","mean Cluster 2","mean Cluster 3",\
                                 "std Cluster 1","std Cluster 2","std Cluster 3"])
df_stats.insert(0,"clinical_scores",NP_scores)
for key in NP_scores:
    neurospycho = df_scores[key].astype(np.float).values
    df = pd.DataFrame()
    y = y_nudast[np.array(np.isnan(neurospycho)==False)]
    score = neurospycho[np.array(np.isnan(neurospycho)==False)]
    score = ((score - score[y==0].mean(axis=0))/score[y==0].std(axis=0))
    df[key] = score
    df["labels"]=labels_cluster[np.array(np.isnan(neurospycho)==False)]
    df_stats.loc[df_stats.clinical_scores==key,"mean Controls"] = round(df[df["labels"]=='Controls'][key].mean(),3)

    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 1"] = round(df[df["labels"]=='Subcortical'][key].mean(),3)
    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 2"] = round(df[df["labels"]=='Cortical'][key].mean(),3)
    df_stats.loc[df_stats.clinical_scores==key,"mean Cluster 3"] = round(df[df["labels"]=='Preserved'][key].mean(),3)

    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 1"] = round(df[df["labels"]=='Subcortical'][key].std(),3)
    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 2"] = round(df[df["labels"]=='Cortical'][key].std(),3)
    df_stats.loc[df_stats.clinical_scores==key,"std Cluster 3"] = round(df[df["labels"]=='Preserved'][key].std(),3)


plt.plot(df_stats["mean Controls"],'o',label = 'Controls', marker='o',markersize=10,ls='--',color= "darkgreen")
plt.plot(df_stats["mean Cluster 1"],'o',label = 'Subcortical', marker='v',markersize=10,ls='--',color= "darkblue")
plt.plot(df_stats["mean Cluster 2"],'o',label = 'Cortical', marker='s',markersize=10,ls='--',color= "firebrick")
plt.plot(df_stats["mean Cluster 3"],'o',label = 'Preserved', marker='d',markersize=10,ls='--',color= "goldenrod")
plt.legend(loc = 'upper center',ncol =4)
plt.ylabel("Z-score")
x = np.arange(5)
NP_scores_legend = ["Sum of Trials 1-5","Short Delay Cued Recall",\
"Long Delay Cued Recall","Learning Slope","Discriminability"]
plt.xticks(x,NP_scores_legend,rotation=60, fontsize=12)
plt.tight_layout()
plt.title("California Verbal Learning Task")


###############################################################################
## Libraries
#import matplotlib.pyplot as plt
#import pandas as pd
#from math import pi
#NP_scores = ["vocabsca",'dstscalc',"sstscalc","lnsscalc","d4prime","lmiscalc",\
#"fpiscalc",'matrxsca',"trailb","wcstpsve"]
#
## Set data
#df = pd.DataFrame({
#'group': ['Cluster 1','Cluster 2','Cluster 3'],
#'WAIS vocab': [df_stats[df_stats["clinical_scores"]=="vocabsca"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="vocabsca"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="vocabsca"]["mean Cluster 3"].values],\
#
#'WMS Digit Span':  [df_stats[df_stats["clinical_scores"]=="dstscalc"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="dstscalc"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="dstscalc"]["mean Cluster 3"].values],\
#
#'WMS Spatial Span': [df_stats[df_stats["clinical_scores"]=="sstscalc"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="sstscalc"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="sstscalc"]["mean Cluster 3"].values],
#
#'WMS LN': [df_stats[df_stats["clinical_scores"]=="lnsscalc"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="lnsscalc"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="lnsscalc"]["mean Cluster 3"].values],\
#
#'CPT dprime': [df_stats[df_stats["clinical_scores"]=="d4prime"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="d4prime"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="d4prime"]["mean Cluster 3"].values],\
#
#'WMS LM':  [df_stats[df_stats["clinical_scores"]=="lmiscalc"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="lmiscalc"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="lmiscalc"]["mean Cluster 3"].values],\
#
#'WMS FP': [df_stats[df_stats["clinical_scores"]=="fpiscalc"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="fpiscalc"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="fpiscalc"]["mean Cluster 3"].values],
#
#'WAIS matrix': [df_stats[df_stats["clinical_scores"]=="matrxsca"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="matrxsca"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="matrxsca"]["mean Cluster 3"].values],\
#
#'Trails B': [df_stats[df_stats["clinical_scores"]=="trailb"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="trailb"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="trailb"]["mean Cluster 3"].values],\
#
#'WCST errors':  [df_stats[df_stats["clinical_scores"]=="wcstpsve"]["mean Cluster 1"].values,\
#               df_stats[df_stats["clinical_scores"]=="wcstpsve"]["mean Cluster 2"].values,\
#                df_stats[df_stats["clinical_scores"]=="wcstpsve"]["mean Cluster 3"].values]
#
#})
#NP_scores_legend = ["WAIS vocab",'WMS Digit Span',"WMS Spatial Span","WMS LN",\
#"CPT dprime", "WMS LM","WMS FP",'WAIS matrix','Trails B','WCST errors',"group"]
##NP_scores_legend = ["WAIS vocab",'WMS Digit Span',"WMS Spatial Span","WMS LN",\
##"CPT dprime", "WMS LM","WMS FP",'WAIS matrix',"group"]
#
#df = df[NP_scores_legend]
#
## number of variable
#categories=list(df)[:-1]
#N = len(categories)
#
## We are going to plot the first line of the data frame.
## But we need to repeat the first value to close the circular graph:
#values0=df.loc[0].drop('group').values.flatten().tolist()
#values0 += values0[:1]
#values0
#values1=df.loc[1].drop('group').values.flatten().tolist()
#values1 += values1[:1]
#values1
#
#values2=df.loc[2].drop('group').values.flatten().tolist()
#values2 += values2[:1]
#values2
#
#
## What will be the angle of each axis in the plot? (we divide the plot / number of variable)
#angles = [n / float(N) * 2 * pi for n in range(N)]
#angles += angles[:1]
#
## Initialise the spider plot
#ax = plt.subplot(111, polar=True)
#
## Draw one axe per variable + add labels labels yet
#plt.xticks(angles[:-1], categories, color='black', size=12)
#
## Draw ylabels
#ax.set_rlabel_position(0)
#plt.yticks([0], ["controls"], color="grey", size=7)
#plt.ylim(-2,3)
#
## Plot data
#ax.plot(angles, values0, linewidth=3, linestyle='solid',label= "Cluster 1")
#ax.plot(angles, values1, linewidth=3, linestyle='solid',label="Cluster 2")
#ax.plot(angles, values2, linewidth=3, linestyle='solid',label="Cluster 3")
#ax.legend(loc=7)
## Fill area

