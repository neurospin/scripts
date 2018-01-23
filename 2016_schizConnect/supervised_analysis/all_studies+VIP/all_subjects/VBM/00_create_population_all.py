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

BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/completed_schizconnect_metaData_1829.csv'
OUTPUT_CSV = os.path.join(BASE_PATH,"population.csv")


INPUT_CSV_COBRE = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/VBM/population.csv"
INPUT_CSV_NMorphCH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/VBM/population.csv"
INPUT_CSV_NUSDAST = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/VBM/population.csv"
INPUT_CSV_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population.csv"


clinic_COBRE = pd.read_csv(INPUT_CSV_COBRE)
clinic_NMorphCH = pd.read_csv(INPUT_CSV_NMorphCH)
clinic_NUSDAST = pd.read_csv(INPUT_CSV_NUSDAST)
clinic_VIP = pd.read_csv(INPUT_CSV_VIP)

#Since VIP is not a SchizConnect database, need to reorganise some infos
clinic_VIP = clinic_VIP[['path_VBM','sex_code','dx','age']]
clinic_VIP["site"] = "vip"
clinic_VIP["sex_num"] = clinic_VIP["sex_code"]
clinic_VIP["dx_num"] = clinic_VIP["dx"]
clinic_VIP = clinic_VIP[['path_VBM','sex_num','dx_num','site','age']]
clinic_VIP['subjectid'] = 'code_vip'





all_clinic = [clinic_COBRE, clinic_NMorphCH, clinic_NUSDAST,clinic_VIP]
pop = pd.concat(all_clinic)
pop = pop[['subjectid',"dx_num","path_VBM","sex_num","site","age"]]
assert pop.shape == (606, 6)

#pop.site.unique()
SITE_MAP = {'MRN': 1, 'NU': 2, "WUSTL" : 3,"vip" : 4}
pop['site_num'] = pop["site"].map(SITE_MAP)
assert sum(pop.site_num.values==1) == 164
assert sum(pop.site_num.values==2) == 80
assert sum(pop.site_num.values==3) == 270
assert sum(pop.site_num.values==4) == 92


assert sum(pop.dx_num.values==0) == 330
assert sum(pop.dx_num.values==1) == 276

assert sum(pop.sex_num.values==0) == 374
assert sum(pop.sex_num.values==1) == 232

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)

#pop[pop.site_num==4].age.mean()
#Out[116]: 34.37954139368672
#
#pop[pop.site_num==3].age.mean()
#Out[117]: 30.581481481481482
#
#pop[pop.site_num==2].age.mean()
#Out[118]: 32.05
#
#pop[pop.site_num==1].age.mean()
#Out[119]: 37.84146341463415
##############################################################################
#