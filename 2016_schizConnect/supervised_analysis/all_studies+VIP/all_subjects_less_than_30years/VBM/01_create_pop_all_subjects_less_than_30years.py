#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:43:27 2017

@author: ad247405
"""
import os
import numpy as np
import pandas as pd
import glob

BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects_less_than_30years'
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






all_clinic = [clinic_COBRE, clinic_NMorphCH, clinic_NUSDAST,clinic_VIP]
pop = pd.concat(all_clinic)
pop = pop[["dx_num","path_VBM","sex_num","site","age"]]
assert pop.shape == (606, 5)


#Only subjects less than 50 years old
pop = pop[pop.age<30]
assert pop.shape == (297, 5)

SITE_MAP = {'MRN': 1, 'NU': 2, "WUSTL" : 3,"vip" : 4}
pop['site_num'] = pop["site"].map(SITE_MAP)


# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
