#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:43:27 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import re
import glob



BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects"
OUTPUT_CSV = os.path.join(BASE_PATH,"population.csv")


INPUT_CSV_COBRE = "/neurospin/brainomics/2016_schizConnect/analysis/COBRE/Freesurfer/population.csv"
INPUT_CSV_NMorphCH = "/neurospin/brainomics/2016_schizConnect/analysis/NMorphCH/Freesurfer/population.csv"
INPUT_CSV_NUSDAST = "/neurospin/brainomics/2016_schizConnect/analysis/NUSDAST/Freesurfer/population.csv"
INPUT_CSV_VIP = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/Freesurfer/population.csv"


clinic_COBRE = pd.read_csv(INPUT_CSV_COBRE)
clinic_NMorphCH = pd.read_csv(INPUT_CSV_NMorphCH)
clinic_NUSDAST = pd.read_csv(INPUT_CSV_NUSDAST)
clinic_VIP = pd.read_csv(INPUT_CSV_VIP)



all_clinic = [clinic_COBRE, clinic_NMorphCH, clinic_NUSDAST,clinic_VIP]
pop = pd.concat(all_clinic)


#pop.site.unique() 
SITE_MAP = {'MRN': 1, 'NU': 2, "WUSTL" : 3,"vip":4}
pop['site_num'] = pop["site"].map(SITE_MAP)
assert sum(pop.site_num.values==1) == 154
assert sum(pop.site_num.values==2) == 73
assert sum(pop.site_num.values==3) == 260
assert sum(pop.site_num.values==4) == 80



assert sum(pop.dx_num.values==0) == 314
assert sum(pop.dx_num.values==1) == 253

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)
