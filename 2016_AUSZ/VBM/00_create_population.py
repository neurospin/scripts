# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:42:15 2016

@author: ad247405
"""

import os
import glob
import pandas as pd

#import proj_classif_config
GROUP_MAP = {1: 1, '2a': 2, '2b': 3, 3 : 0}
GENDER_MAP = {'F': 0, 'M': 1}

#1 TSA
#2a scz-asd
#2b: schizophrenia
#3 controls

BASE_PATH = '/neurospin/brainomics/2016_AUSZ'
INPUT_CLINIC_FILENAME = '/neurospin/brainomics/2016_AUSZ/documents/Tableau_IRM_AUSZ_GM_16_12.xlsx'
INPUT_DATA = '/neurospin/brainomics/2016_AUSZ/data/'
OUTPUT_CSV = os.path.join(BASE_PATH,"results","VBM","population.csv")


#AUSZpopulation file
##############################################################################
# Read clinic data
clinic = pd.read_excel(INPUT_CLINIC_FILENAME,header = (1))
assert  clinic.shape == (136, 11)
#Take into account error of codes as notified by Gilles Martinez

#First error : GM120123 --> GO120133
row_index = clinic['IRM.1'] == 'GM120133'
clinic.loc[row_index,"IRM.1"] =  'GO120133'

#2nd error : SA120160 --> SA120159
row_index = clinic['IRM.1'] == 'SA120159'
clinic.loc[row_index,"IRM.1"] =  'SA120160'

#3rd error :  SB1201I8  --> SB120158 
row_index = clinic['IRM.1'] == 'SB1201I8'
clinic.loc[row_index,"IRM.1"] =  'SB120158'

#4th error :  ME150271   --> ME150371 
row_index = clinic['IRM.1'] == 'ME150271'
clinic.loc[row_index,"IRM.1"] =  'ME150371'

#5th error :  KD150466  --> KH150466
row_index = clinic['IRM.1'] == 'KD150466'
clinic.loc[row_index,"IRM.1"] =  'KH150466'

#6th error :  RE160510  --> RE1606510 
row_index = clinic['IRM.1'] == 'RE160510'
clinic.loc[row_index,"IRM.1"] =  'RE1606510'

#7th error :  
row_index = clinic['Ausz'] == 'AZ-RH-01-105'
clinic.loc[row_index,"IRM.1"] =  'RH150462'

#8th error : not spotted by Gilles 
row_index = clinic['IRM.1'] == 'LE160192'
clinic.loc[row_index,"IRM.1"] =  'LE160692'

#9th error : not spotted by Gilles 
row_index = clinic['IRM.1'] == 'RC160671'
clinic.loc[row_index,"IRM.1"] =  'RC160676'

assert  clinic.shape == (136, 11)


subjects = list()
paths = glob.glob(INPUT_DATA+"/*/*/*/T1_VBM/mwrc1T1.nii")
for i in range(len(paths)):
    subjects.append(os.path.split(os.path.split(os.path.split(os.path.split(paths[i])[0])[0])[0])[1]) 

input_subjects_vbm = pd.DataFrame(subjects, columns=["IRM.1"])
input_subjects_vbm["path_VBM"] = paths
assert input_subjects_vbm.shape == (130, 2)                        

#Remove the 4 images that are not AUSZ (according to Gilles mail)
input_subjects_vbm = input_subjects_vbm[input_subjects_vbm["IRM.1"] != "SA130280"]
input_subjects_vbm = input_subjects_vbm[input_subjects_vbm["IRM.1"] != "CF140297"]
input_subjects_vbm = input_subjects_vbm[input_subjects_vbm["IRM.1"] != "BM130180"]
input_subjects_vbm = input_subjects_vbm[input_subjects_vbm["IRM.1"] != "CT130238"]

assert input_subjects_vbm.shape == (126, 2)



# intersect with subject with image
clinic = clinic.merge(input_subjects_vbm, on="IRM.1")
pop = clinic[["IRM.1","Groupe", "Sexe", "Âge", "path_VBM"]]
assert pop.shape == (126, 5) 

pop = pop[pop["Groupe"]!="exclu"]
assert pop.shape == (123, 5) 

# Map group
pop['group.num'] = pop["Groupe"].map(GROUP_MAP)
pop['sex.num'] = pop["Sexe"].map(GENDER_MAP)
# Save population information
pop.to_csv(OUTPUT_CSV, encoding='utf-8' ) 
##############################################################################
