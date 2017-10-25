#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:35:27 2017

@author: ad247405
"""
import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns, matplotlib.pyplot as plt
import scipy.stats
import nibabel
import scipy.stats

INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/X.npy"

INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/data/mean_centered_by_site_all/y.npy"

POPULATION_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv"

pop = pd.read_csv(POPULATION_CSV)
age = pop["age"].values


X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y).ravel()
site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/site.npy")

X_vip = X[site==4,:]
y_vip = y[site==4]
X_vip_scz = X_vip[y_vip==1,2:]
assert X_vip_scz.shape == (39, 125959)
age_scz = age[y_vip==1]
age_con = age[y_vip==0]



MASK_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/mask.nii"
babel_mask  = nibabel.load(MASK_PATH)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/\
results/enetall_all+VIP_all/5cv/refit/refit/enettv_0.1_0.1_0.8"
beta = np.load(os.path.join(WD,"beta.npz"))['arr_0'][3:]


CLUSTER_LABELS = "/neurospin/brainomics/2016_schizConnect/analysis/\
all_studies+VIP/VBM/all_subjects/results/enetall_all+VIP_all/5cv/refit/refit/\
enettv_0.1_0.1_0.8/weight_map_clust_labels.nii.gz"


labels_img  = nibabel.load(CLUSTER_LABELS)
labels_arr = labels_img.get_data()
labels_flt = labels_arr[mask_bool]


N = X_vip_scz.shape[0]
N_all = X_vip.shape[0]

# Extract a single score for each cluster
K = len(np.unique(labels_flt))  # nb cluster
scores_vip_scz = np.zeros((N, K))
scores_vip_all = np.zeros((N_all, K))

K_interest = [18,14,33,20,4,25,23,22,15,41]

for k in range (K):
    mask = labels_flt == k
    print("Cluster:",k, "size:", mask.sum())
    scores_vip_scz[:, k] = np.dot(X_vip_scz[:, mask], beta[mask]).ravel()
    scores_vip_all[:, k] = np.dot(X_vip[:, mask], beta[mask]).ravel()




INPUT_SCORES = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS"


SAPS_vip = np.load(os.path.join(INPUT_SCORES,"SAPS_vip.npy"))
SANS_vip = np.load(os.path.join(INPUT_SCORES,"SANS_vip.npy"))

#remove subejct with no available scores
scores_vip_scz =  scores_vip_scz[np.logical_not(np.isnan(SANS_vip))]
age = age[np.logical_not(np.isnan(SANS_vip))]
SAPS_vip = SAPS_vip[np.logical_not(np.isnan(SANS_vip))]
SANS_vip = SANS_vip[np.logical_not(np.isnan(SANS_vip))]


dose = dose[np.logical_not(np.isnan(SANS_vip))]

assert SAPS_vip.shape == SANS_vip.shape == (35,)

 # PLOT ALL CORRELATIONW WITH SAPS

K_interest = [18,14,33,20,4,25,23,22,15,41]


#Correlation with age
##############################################################################
for i in (K_interest):
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],age)
    print (p)
##############################################################################
#Correlation with scores
##############################################################################

for i in (K_interest):
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],SAPS_vip)
    print (p)

for i in (K_interest):
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],SANS_vip)
    print (p)
##############################################################################

#Plot PANSS correlation
for i in K_interest:
    fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],SAPS_vip)
    ax1.plot(scores_vip_scz[:,i] ,SAPS_vip,'o')
    ax1.set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    ax1.set_xlabel('Score on cluster %s' %(i))
    ax1.set_ylabel('SAPS score')
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],SANS_vip)
    ax2.plot(scores_vip_scz[:,i] ,SANS_vip,'o')
    ax2.set_title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    ax2.set_xlabel('Score on cluster %s' %(i))
    ax2.set_ylabel('SANS score')
    fig.tight_layout()
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/vip_cluster%s" %(i))

#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_vip_all[:,i],age)
    plt.figure()
    plt.plot(scores_vip_all[y_vip==0,i] ,age_con,'o',label = "controls")
    plt.plot(scores_vip_all[y_vip==1,i] ,age_scz,'o',label = "patients")
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('age')
    plt.tight_layout()
    plt.legend(fontsize = 15,loc = "upper left")
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/age_corr/vip_cluster_age%s" %(i))

#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],age)
    plt.figure()
    plt.plot(scores_vip_scz[:,i] ,age,'o')
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('age')
    plt.tight_layout()
    plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/clinic_correlations/age_corr/vip_cluster_age%s" %(i))






#Antipsychotic - Age of onset
##############################################################################
age_onset_antipsychotic = pop["TTT_VIE_NEURO_AAO"].values[y_vip==1]
age_onset_atypical_antipsychotic = pop["TTT_VIE_ANTIPSY_AAO"].values[y_vip==1]

df = pd.DataFrame()
df["age_onset_antipsychotic"] = age_onset_antipsychotic
df["age_onset_atypical_antipsychotic"] = age_onset_atypical_antipsychotic
df = df.fillna(999)
df = df.replace(to_replace = "NC",value = 999)
df = df.astype(float)
df["age_onset"] = df[["age_onset_antipsychotic","age_onset_atypical_antipsychotic"]].min(axis=1)

age_onset = df["age_onset"].values
scores_vip_scz = scores_vip_scz[age_onset!= 999,:]
age_onset = age_onset[age_onset!= 999]


#Plot age correlations
for i in K_interest:
    corr,p = scipy.stats.pearsonr(scores_vip_scz[:,i],age_onset)
    plt.figure()
    plt.plot(scores_vip_scz[:,i] ,age_onset,'o')
    plt.title("Pearson' s correlation = %.02f \n p = %.01e" % (corr,p),fontsize=12)
    plt.xlabel('Score on cluster %s' %(i))
    plt.ylabel('age')
    plt.tight_layout()

##############################################################################


#Antipsychotic - dose received per day
##############################################################################
pop = pop[pop["diagnostic"]==3]


antipsychotic = ['CLOZAPINE','ARIPIPRAZOLE','RISPERIDONE','OLANZAPINE','LEVOMEPROMAZINE',"CYAMEMAZINE",\
                 "VALPROMIDE","HALOPERIDOL","LOXAPINE","AMISULPRIDE"]

antipsychotic_dose = [100/50,100/7.5,100/2,100/5,'LEVOMEPROMAZINE',"CYAMEMAZINE",\
                 "VALPROMIDE",100/2,100/10,100/10]

pop['MEDPOSO1'] = pop['MEDPOSO1'].str.replace(pat = "MG/J",repl = "")
pop['MEDPOSO1'] = pop['MEDPOSO1'].str.replace(pat = "MG",repl = "")
pop[pop['MEDPOSO1'] == "50µ/14J"] = 0.05/14
pop[pop['MEDPOSO1'] == "75/4SEM"] = 75/28

for i in range (len(pop)):
    print(i)
    curr= pop[pop.index == i]
    print(curr['MEDLIATC1'])
    for idx,  med in enumerate(antipsychotic):
        print(med)
        print(idx)

        curr = pop.replace(to_replace = "MG/J",value="")
        curr[curr['MEDLIATC1']==med]['MEDPOSO1'] * antipsychotic_dose[idx]


pop['MEDPOSO1'].map(lambda x: x.lstrip('MG').rstrip('aAbBcC'))


Clozapine*100/50
        curr[curr['MEDLIATC1']=='ARIPIPRAZOLE']['MEDPOSO1'] *100/7.5
        curr[curr['MEDLIATC1']=='RISPERIDONE']['MEDPOSO1'] *100/2
        curr[curr['MEDLIATC1']=='OLANZAPINE']['MEDPOSO1'] *100/5i=1
        curr[curr['MEDLIATC1']=='LEVOMEPROMAZINE']['MEDPOSO1'] ???
        curr[curr['MEDLIATC1']=='CYAMEMAZINE']['MEDPOSO1']  ??
        curr[curr['MEDLIATC1']=='LOXAPINE']['MEDPOSO1'] *100/10
        curr[curr['MEDLIATC1']=='AMISULPRIDE']['MEDPOSO1'] *100/10



#LEVOMEPROMAZINE: antipsychotic eqOlan??
#TROPATEPINE ??
#CLONAZEPAM,LORAZEPAM PAROXETINE = anxiolytique
#ZOPICLONE,ALIMEMAZINE sommeil
#VALPROIC ACID thymorégulateurs
#CARBAMAZEPINE
#ESCITALOPRAM depression
#BUSPIRONE anxiolytique
#TROPATEPINE anxiolytiaue
#ESCITALOPRAM
#'OXAZEPAM'
pop['MEDLIATC1'].unique()
pop['MEDLIATC2'].unique()
pop['MEDLIATC3'].unique()
pop['MEDLIATC4'].unique()
pop['MEDLIATC5'].unique()
pop['MEDLIATC6'].unique()
pop['MEDLIATC7'].unique()
pop['MEDLIATC8'].unique()