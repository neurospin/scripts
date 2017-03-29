#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 10:08:34 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import glob

BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/completed_schizconnect_metaData_1829.csv'
OUTPUT_CSV = os.path.join(BASE_PATH,"population.csv")


INPUT_CSV_COBRE = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
INPUT_CSV_NMorphCH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
INPUT_CSV_NUSDAST = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"


clinic_COBRE = pd.read_csv(INPUT_CSV_COBRE)
clinic_NMorphCH = pd.read_csv(INPUT_CSV_NMorphCH)
clinic_NUSDAST = pd.read_csv(INPUT_CSV_NUSDAST)



all_clinic = [clinic_COBRE, clinic_NMorphCH, clinic_NUSDAST]
pop = pd.concat(all_clinic)

assert pop.shape == (514, 9)

#pop.site.unique() 
SITE_MAP = {'MRN': 1, 'NU': 2, "WUSTL" : 3}
pop['site_num'] = pop["site"].map(SITE_MAP)
assert sum(pop.site_num.values==1) == 164
assert sum(pop.site_num.values==2) == 80
assert sum(pop.site_num.values==3) == 270



assert sum(pop.dx_num.values==0) == 277
assert sum(pop.dx_num.values==1) == 237

assert sum(pop.sex_num.values==0) == 323
assert sum(pop.sex_num.values==1) == 191

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)


##############################################################################
#Create list of subejcts for Preprocessing (Global template)


OUTPUT_T1_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_T1.txt"
file = open(OUTPUT_T1_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/average_T1.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/average_T1.nii\n")      
file.close()


OUTPUT_WM_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_WM.txt"
file = open(OUTPUT_WM_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/c2average_T1.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/c2average_T1.nii\n")
file.close()
    
OUTPUT_DARTEL_WM_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_WM_dartel_imported.txt"
file = open(OUTPUT_DARTEL_WM_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/rc2average_T1.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/rc2average_T1.nii\n")
file.close()

    
OUTPUT_DARTEL_GM_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_GM_dartel_imported.txt"
file = open(OUTPUT_DARTEL_GM_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/rc1average_T1.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/rc1average_T1.nii\n")    
file.close()

    
OUTPUT_GM_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_GM.txt"
file = open(OUTPUT_GM_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/c1average_T1.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/c1average_T1.nii\n")    
file.close()    
    
    
OUTPUT_FLOWFIELDS_LIST = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/all_subjects/preproc/list_subjects/list_flowfields.txt"
file = open(OUTPUT_FLOWFIELDS_LIST, 'w')
for i in range(pop.shape[0]):
    print(os.path.split(pop.path_VBM.values[i])[0] + "/u_rc1average_T1_Template.nii")
    file.write(os.path.split(pop.path_VBM.values[i])[0] + "/u_rc1average_T1_Template.nii\n")      
file.close()    