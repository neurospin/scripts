
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
import sklearn


#1) Extract NUDAST PANSS scores
###############################################################################
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NUSDAST_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
#PROBLEME multiple scores for each subject!!

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr = curr[curr.visit == most_recent_visit]
    current_SAPS = curr[curr.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.int64).values
    current_SANS = curr[curr.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.int64).values
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
    if len(current_SAPS) != 0:
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()

#investigate distribution of SAPS and SANS scores across SCZ population
SAPS_nudast =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_nudast =  pop[pop.dx_num ==1].SANS.astype(np.float).values
assert SAPS_nudast.shape == SANS_nudast.shape == (118,)

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nudast.npy",SAPS_nudast)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nudast.npy",SANS_nudast)
###############################################################################



#2) Extract  VIP PANSS scores
###############################################################################
INPUT_VIP_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population.csv"
clinic = pd.read_csv(INPUT_VIP_POPULATION)
clinic = clinic[clinic.diagnostic==3.0]
SANS_vip = clinic['PANSS_NEGATIVE'].values
SAPS_vip = clinic['PANSS_POSITIVE'].values
assert SAPS_vip.shape == SANS_vip.shape == (39,)

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_vip.npy",SAPS_vip)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_vip.npy",SANS_vip)


#3) Extract NMorphCH PANSS scores
###############################################################################
INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_NMorphCH_assessmentData_1829.csv"
INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"

clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
clinic.fillna(0, inplace=True)
pop = pd.read_csv(INPUT_POPULATION)
pop["SAPS"] =  "NaN"
pop["SANS"] =  "NaN"
for s in pop.subjectid:
    curr = clinic[clinic.subjectid ==s]
    most_recent_visit = curr.visit.unique()[-1]
    curr = curr[curr.visit == most_recent_visit]
    current_SAPS = curr[curr.assessment_description == "Scale for the Assessment of Positive Symptoms"].question_value.astype(np.int64).values
    current_SANS = curr[curr.assessment_description == "Scale for the Assessment of Negative Symptoms"].question_value.astype(np.int64).values
    if len(current_SANS) != 0:
        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
    if len(current_SAPS) != 0:
        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()

#investigate distribution of SAPS and SANS scores across SCZ population
SAPS_nmorphch =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
SANS_nmorphch =  pop[pop.dx_num ==1].SANS.astype(np.float).values
assert SAPS_nmorphch.shape == SANS_nmorphch.shape == (42,)

np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nmorphch.npy",SAPS_nmorphch)
np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nmorphch.npy",SANS_nmorphch)
###############################################################################

#4) Extract COBRE PANSS scores
################################################################################
#INPUT_CLINIC_FILENAME = "/neurospin/abide/schizConnect/data/schizconnect_COBRE_assessmentData_1829.csv"
#INPUT_POPULATION = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
#
#clinic = pd.read_csv(INPUT_CLINIC_FILENAME)
##clinic.fillna(0, inplace=True)
#pop = pd.read_csv(INPUT_POPULATION)
#pop["SAPS"] =  "NaN"
#pop["SANS"] =  "NaN"
#
#pd.unique(clinic[clinic.assessment_description == "Positive and Negative Symptom Scale"].question_value)
#
#
#for s in pop.subjectid:
#    curr = clinic[clinic.subjectid ==s]
#    most_recent_visit = curr.visit.unique()[-1]
#    curr = curr[curr.visit == most_recent_visit]
#    current_PANSS = curr[curr.assessment_description == "Positive and Negative Symptom Scale"].question_value.values
#    if len(current_SANS) != 0:
#        pop.loc[pop.subjectid ==s,"SAPS"] = current_SAPS.sum()
#    if len(current_SAPS) != 0:
#        pop.loc[pop.subjectid ==s,"SANS"] = current_SANS.sum()
#
##investigate distribution of SAPS and SANS scores across SCZ population
#SAPS_nmorphch =  pop[pop.dx_num ==1].SAPS.astype(np.float).values
#SANS_nmorphch =  pop[pop.dx_num ==1].SANS.astype(np.float).values
#assert SAPS_nmorphch.shape == SANS_nmorphch.shape == (118,)
#
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SAPS_nmorphch.npy",SAPS_nmorphch)
#np.save("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/SANS_nmorphch.npy",SANS_nmorphch)
################################################################################
SAPS_vip = SAPS_vip[np.logical_not(np.isnan(SANS_vip))]
SANS_vip = SANS_vip[np.logical_not(np.isnan(SANS_vip))]
SAPS_nmorphch = SAPS_nmorphch[np.logical_not(np.isnan(SANS_nmorphch))]
SANS_nmorphch = SANS_nmorphch[np.logical_not(np.isnan(SANS_nmorphch))]


plt.figure()
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.hist(SAPS_nudast)
ax1.set_title("SAPS score - NUSDAST site",fontsize=12)
ax1.set_xlabel('SAPS score')
ax2.hist(SANS_nudast)
ax2.set_title("SANS score - NUSDAST site",fontsize=12)
ax2.set_xlabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/nudast_scores.png")

plt.figure()
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.hist(SAPS_vip)
ax1.set_title("SAPS score - VIP site",fontsize=12)
ax1.set_xlabel('SAPS score')
ax2.hist(SANS_vip)
ax2.set_title("SANS score - VIP site",fontsize=12)
ax2.set_xlabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/vip_scores.png")

plt.figure()
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.hist(SAPS_nmorphch)
ax1.set_title("SAPS score - NMorphCH site",fontsize=12)
ax1.set_xlabel('SAPS score')
ax2.hist(SANS_nmorphch)
ax2.set_title("SANS score - NMorphCH site",fontsize=12)
ax2.set_xlabel('SANS score')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/scores_PANSS/nmorph_scores.png")
