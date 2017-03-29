#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:04:05 2017

@author: ad247405
"""

import os
import numpy as np
import pandas as pd
import re
import glob


#BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/VIP"
#INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/analysis/VIP/sujets_series.xls'
#OUTPUT_CSV = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/population.csv"
#
#clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
#clinic = clinic[clinic.diagnostic !=2]
#clinic = clinic[clinic.diagnostic !=4]
#clinic = clinic[clinic.diagnostic.isnull()!= True]
#clinic["path_subject"] = "/neurospin/lnao/Pdiff/josselin/ellen/VIP/subjects/"+  (clinic["nip"]).values
#clinic["path_t1"] = (clinic["path_subject"]).values + "/fMRI/acquisition1/"+ clinic["nip"].values +".nii"
#list_subjects = clinic.nip.values
#clinic.to_csv(OUTPUT_CSV, index=False)
#
#
#
#
#OUTPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/data" 
##script to find out missing images 
#for p in clinic["path_t1"].values:
#    print(p)
#    new_path = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/data" 
#    cmd = "cp " + p + " " + new_path
#    a = os.system(cmd)
#    if a!=0:
#        print("ISSUE")
    
##################################################################################################


import os
import numpy as np
import pandas as pd
import glob

INPUT_DATA = "/neurospin/brainomics/2016_schizConnect/analysis/VIP/data"
BASE_PATH = '/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM'
INPUT_CLINIC_FILENAME =  '/neurospin/brainomics/2016_schizConnect/analysis/VIP/sujets_series.xls'
OUTPUT_CSV = os.path.join(BASE_PATH,"population.csv")





# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME)
clinic = clinic[clinic.diagnostic !=2]
clinic = clinic[clinic.diagnostic !=4]
clinic = clinic[clinic.diagnostic.isnull()!= True]


# Read subjects with image
subjects = list()
path_subjects= list()
paths = glob.glob(os.path.join(INPUT_DATA,"mwc1*.nii"))
for i in range(len(paths)):
    path_subjects.append(paths[i])
    subjects.append(os.path.split(paths[i])[1][4:-4])


    
pop = pd.DataFrame(subjects, columns=["nip"])
pop["path_VBM"] = path_subjects
assert pop.shape == (92, 2)                        


pop = clinic.merge(pop, on = "nip")
          
# Map group
DX_MAP = {1.0: 0, 3.0: 1}
SEX_MAP = {1.0: 0, 2.0: 1.0}
pop['dx'] = pop["diagnostic"].map(DX_MAP)
pop['sex_code'] = pop["sexe"].map(SEX_MAP)


assert sum(pop.dx.values==0) == 53
assert sum(pop.dx.values==1) == 39
assert sum(pop.sex_code.values==0) == 51
assert sum(pop.sex_code.values==1) == 41

from datetime import datetime, timedelta

pd.to_datetime(pop["ddn"])
pd.to_datetime(pop["acq date"],format='%Y%m%d')
pop.age_days = pd.to_datetime(pop["acq date"],format='%Y%m%d') - pd.to_datetime(pop["ddn"])
pop['age'] = pop.age_days/ timedelta(days=365)

# Save population information
pop.to_csv(OUTPUT_CSV, index=False)

##############################################################################
