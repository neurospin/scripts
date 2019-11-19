#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:41:12 2019

@author: ai258328
"""

import os
import numpy as np
import glob
import pandas as pd
#import nibabel
#import brainomics.image_atlas
import shutil
#import mulm
#import sklearn
import re
#from nilearn import plotting
#import matplotlib.pyplot as plt
#import scipy, scipy.ndimage
import xml.etree.ElementTree as ET
import re

BASE_PATH_icaar = '/neurospin/psy_sbox/start-icaar-eugei/derivatives/cat12/stats'
INPUT_CSV_icaar = '/neurospin/psy_sbox/start-icaar-eugei/phenotype'
BASE_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague/derivatives/cat12/stats'
INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'
OUTPUT_PATH = '/neurospin/psy_sbox/analysis/201906_start-icaar-eugei-schizconnect-vip-prague-bsnip_cat12-roi_predict-transition/data'

"""
NOTE IMPORTANTE: je me suis assuré que l'ordre des ROIs et des phénotypes soit toujours le même, avant et après concaténation des cohortes
"""

"""
MAKE DATASET ICAAR-EUGEI-START
"""

# Normalize the ICAAR-EUGEI-START data, using total intracranial volume

ROIs = pd.read_csv(os.path.join(BASE_PATH_icaar, 'cat12_rois_Vgm.tsv'),sep='\t')
tissue_vol = pd.read_csv(os.path.join(BASE_PATH_icaar, 'cat12_tissues_volumes.tsv'),sep='\t')
assert ROIs.shape == (171, 143)
tissue_vol['norm_ratio'] = 1500 / tissue_vol.tiv
for i in range(ROIs.shape[0]):
    ROIs.iloc[i,1:] *= tissue_vol.norm_ratio[i]
    
# Homogeneize and merge ICAAR-EUGEI and START phenotypes
    
icaar_eugei = pd.read_csv(os.path.join(INPUT_CSV_icaar,'clinic_icaar_201907.tsv'), sep='\t')
start = pd.read_csv(os.path.join(INPUT_CSV_icaar,'clinic_start_201907.tsv'), sep='\t')
# icaar renommé pour correspondre avec start
icaar_eugei['clinical_status'] = np.nan
for i in range(icaar_eugei.shape[0]):
    if icaar_eugei.Baseline_Status[i] == 'UHR' and icaar_eugei.Conversion[i] == 'yes':
        icaar_eugei.clinical_status[i] = 'UHR-C'
    elif icaar_eugei.Baseline_Status[i] == 'UHR' and icaar_eugei.Conversion[i] == 'no':
        icaar_eugei.clinical_status[i] = 'UHR-NC'
    elif icaar_eugei.Baseline_Status[i] == 'NUHR' and icaar_eugei.Conversion[i] == 'yes':
        icaar_eugei.clinical_status[i] = 'Non-UHR-C'
    elif icaar_eugei.Baseline_Status[i] == 'NUHR' and icaar_eugei.Conversion[i] == 'no':
        icaar_eugei.clinical_status[i] = 'Non-UHR-NC'
    elif icaar_eugei.Baseline_Status[i] == 'Psychotic':
        icaar_eugei.clinical_status[i] = 'Psychotic'
    elif icaar_eugei.Baseline_Status[i] == 'UHR' and pd.isna(icaar_eugei.Conversion[i]):
        icaar_eugei.clinical_status[i] = 'UHR-NaN'
    elif pd.isna(icaar_eugei.Baseline_Status[i]) and pd.isna(icaar_eugei.Conversion[i]):
        icaar_eugei.clinical_status[i] = 'NaN'
# âges de START reformatés
regex = re.compile(".([0-9][0-9]).")
start.iloc[:,3] = [regex.findall(s)[0] for s in start.iloc[:,3]]
for i in range(start.shape[0]):
    start.age.iloc[i] = float(start.age[i])

df_icaar = icaar_eugei[['participant_id','age','sex','clinical_status', 'medication', 'cannabis_last_month', 'tobacco_last_month',
       'BPRS', 'PANSS_total', 'PANSS_positive', 'PANSS_negative',
       'PANSS_psychopatho', 'PANSS_desorganisation', 'SANS', 'SAPS', 'MADRS',
       'SOFAS', 'NSS']]
df_icaar['irm'] = 'M0'

df_start = start[['participant_id','age','sex','clinical_status','medication',
       'cannabis_last_month', 'tobacco_last_month', 'BPRS', 'PANSS_total',
       'PANSS_positive', 'PANSS_negative', 'PANSS_psychopatho',
       'PANSS_desorganisation', 'SANS', 'SAPS', 'MADRS', 'SOFAS', 'NSS','irm']]

icaar_eugei_start = pd.concat([df_icaar, df_start])

assert icaar_eugei_start.shape == (170, 19)

# Merge with the normalized ROIs file

dataset = pd.merge(icaar_eugei_start, ROIs, on='participant_id', how='left')
assert dataset.shape == (170, 161)
dataset.to_csv(os.path.join(OUTPUT_PATH,'norm_dataset_cat12_ICAAR_EUGEI_START.tsv'), sep='\t', index=False)

# Create X and y numpy arrays for testing conversion AMONG UHR (Non-UHR excluded)

conversion_dataset = dataset[pd.notnull(dataset.clinical_status)] # 1 nan exclu
conversion_dataset = dataset[pd.notnull(dataset.clinical_status)] # 1 nan exclu
assert conversion_dataset[conversion_dataset.clinical_status == 'UHR-C'].shape == (32, 161) # 32 converteurs en tout en comptant les M0 et MF
assert conversion_dataset[(conversion_dataset.clinical_status == 'UHR-C') & (conversion_dataset.irm == 'M0')].shape == (27, 161) # mais 27 converteurs à M0
assert conversion_dataset[conversion_dataset.clinical_status == 'UHR-NC'].shape == (64, 161) # 64 UHR non converteurs en tout en comptant M0 et MF
assert conversion_dataset[(conversion_dataset.clinical_status == 'UHR-NC') & (conversion_dataset.irm == 'M0')].shape == (53, 161) # mais 53 UHR-NC à M0

conversion_UHR_data = conversion_dataset[(conversion_dataset.clinical_status.isin(['UHR-C','UHR-NC'])) & (conversion_dataset.irm == 'M0')]
assert conversion_UHR_data.shape == (80, 161)

conversion_UHR_data.clinical_status = conversion_UHR_data.clinical_status.map({'UHR-C':1,'UHR-NC':0})

array_volume = np.array(conversion_UHR_data.iloc[:,np.r_[19:161]]) 
array_conversion = np.array(conversion_UHR_data.clinical_status)
assert array_volume.shape == (80, 142)
assert array_conversion.shape == (80,)
np.save(os.path.join(OUTPUT_PATH, 'X_icaar_eugei_start_ROIs.npy'), array_volume)
np.save(os.path.join(OUTPUT_PATH, 'y_icaar_eugei_start_Clinical_Status.npy'), array_conversion)

# rajouter le site
conversion_UHR_data['site'] = 'nan'
for subject in conversion_UHR_data.participant_id:
    if subject.startswith('START'):
        conversion_UHR_data.site[conversion_UHR_data.participant_id == subject] = 'Sainte-Anne'
    else:
        conversion_UHR_data.site[conversion_UHR_data.participant_id == subject] = 'ICM'

array_site = np.array(conversion_UHR_data.site)        
assert array_site.shape == (80,)
np.save(os.path.join(OUTPUT_PATH, 'sites_icaar_eugei_start.npy'), array_site)

# rajouter le sexe
conversion_UHR_data.sex = conversion_UHR_data.sex.map({'F':1.0,'H':0.0,'M':0.0})
array_sex = np.array(conversion_UHR_data.sex)        
assert array_sex.shape == (80,)
np.save(os.path.join(OUTPUT_PATH, 'sex_icaar_eugei_start.npy'), array_sex)

# target dataset from icaar eugei start
array_UHR_age = np.array(conversion_UHR_data.age)
assert array_UHR_age.shape == (80,)
np.save(os.path.join(OUTPUT_PATH, 'age_icaar_eugei_start.npy'), array_UHR_age)

# MAKE LONGITUDINAL START DATASET
#
#dataset = pd.read_csv('/neurospin/psy_sbox/analysis/201906_start-icaar-eugei-schizconnect-vip-prague-bsnip_cat12-roi_predict-transition/data/norm_dataset_cat12_ICAAR_EUGEI_START.tsv', sep='\t')
#long_pheno = pd.read_csv('/neurospin/psy_sbox/start-icaar-eugei/phenotype/clinic_start_longitudinal_201906.ods', sep='\t')
#dataset = dataset[dataset.participant_id == long_pheno.participant_id]
#del long_pheno['age']
#del long_pheno['clinical_status'] 
#del long_pheno['weight']
#del long_pheno['sex']
#del long_pheno['dcm_name']
#del long_pheno['sequence']
#long_pheno.irm = long_pheno.irm.map({'M0':'M0','M12':'MF'})
#long_data = pd.merge(long_pheno, dataset)
#assert long_data.shape == (28, 149)
#
#array_v = np.array(long_data.iloc[:,np.r_[7:149]])
#array_conversion = np.array(long_data.clinical_status)
#array_sex = np.array(long_data.sex)   
#array_age = np.array(long_data.age)
#assert array_v.shape == (28, 142)
#assert array_conversion.shape == (28,)
#assert array_sex.shape == (28,)
#assert array_age.shape == (28,)
#np.save(os.path.join(OUTPUT_PATH, 'X_longitudinal_start_ROIs.npy'), array_v)
#np.save(os.path.join(OUTPUT_PATH, 'y_longitudinal_start_Clinical_Status.npy'), array_conversion)
#np.save(os.path.join(OUTPUT_PATH, 'sex_longitudinal_start.npy'), array_sex)
#np.save(os.path.join(OUTPUT_PATH, 'age_longitudinal_start.npy'), array_age)

""" 
Create the SchizConnect-PRAGUE-BSNIP schizophrenia dataset
"""

# Normalize the SchizConnect and PRAGUE data, using total intracranial volume
ROIs_SC_PR = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'cat12_rois_Vgm.tsv'),sep='\t')
tissue_vol_SC_PR = pd.read_csv(os.path.join(BASE_PATH_schizconnect, 'cat12_tissues_volumes.tsv'),sep='\t')
assert ROIs_SC_PR.shape == (738, 143)
tissue_vol_SC_PR['norm_ratio'] = 1500 / tissue_vol_SC_PR.tiv
for i in range(ROIs_SC_PR.shape[0]):
    ROIs_SC_PR.iloc[i,1:] *= tissue_vol_SC_PR.norm_ratio[i]
    
# Merge with the phenotype file for SchizConnect
phenotype_SC = pd.read_csv(INPUT_CSV_schizconnect, sep='\t')
phenotype_SC.shape   #phenotype_SC.sex_num
dataset_SC = pd.merge(phenotype_SC, ROIs_SC_PR, on='participant_id', how='left')
assert dataset_SC.shape == (606, 148)
len(list(dataset_SC))
#'participant_id',
# 'dx_num',
# 'path',
# 'sex_num',
# 'site',
# 'age',

# Merge with the phenotype file for PRAGUE
phenotype_PR = pd.read_csv(INPUT_CSV_prague, sep='\t')
phenotype_PR.shape # (133, 6)
# reorganize so that pd.concat doesn't reorder
cols = list(phenotype_PR)
cols.index('dx_num') == 1
cols.index('age') == 4
cols.index('participant_id') == 0
cols.index('sex_num') == 2
cols.index('site') == 3
cols.index('path') == 5
reorder = [cols[0], cols[1], cols[5], cols[2], cols[3], cols[4]]
phenotype_PR = phenotype_PR[reorder]
dataset_PR = pd.merge(phenotype_PR, ROIs_SC_PR, on='participant_id', how='left')
assert dataset_PR.shape == (133, 148)

 # Concatenate SchizConnect and PRAGUE
dataset_SC_PR = pd.concat([dataset_SC, dataset_PR])
assert dataset_SC_PR.shape == (739, 148)
# il y a un sujet en plus dans WUSTL dont on a le phénotype mais pas les images dans ROIs_SC_PR
dataset_SC_PR.participant_id[dataset_SC_PR.participant_id.isin(ROIs_SC_PR.participant_id) == False] # NM1078
dataset_SC_PR[dataset_SC_PR.participant_id == 'NM1078']
#      age  dx_num  l3thVen  l4thVen  ...    rThaPro  rVenVen  sex_num   site
#413  22.0       1      NaN      NaN  ...        NaN      NaN      0.0  WUSTL
dataset_SC_PR = dataset_SC_PR.dropna()
assert dataset_SC_PR.shape == (738, 148)

dataset_SC_PR.to_csv(os.path.join(OUTPUT_PATH,'norm_dataset_cat12_SCHIZCONNECT_VIP_PRAGUE.tsv'), sep='\t')

"""
Extract the schizophrenia data from BSNIP
"""


###################
# TO BE DONE
###################


""" 
Create the SchizConnect/PRAGUE/VIP/BSNIP/BioBD Controls training dataset for prediction of age
"""
# select the controls from SchizConnect
controls_SC_PR = dataset_SC_PR[dataset_SC_PR.dx_num == 0]
assert controls_SC_PR.shape == (420, 148)

# Normalize the BSNIP data, using total intracranial volume
ROIs_BSNIP = pd.read_csv('/neurospin/psy/bsnip1/derivatives/cat12/stats/cat12_rois_Vgm.tsv',sep='\t')
tissue_vol = pd.read_csv('/neurospin/psy/bsnip1/derivatives/cat12/stats/cat12_tissues_volumes.tsv',sep='\t')
assert ROIs_BSNIP.shape == (1042, 143)
tissue_vol['norm_ratio'] = 1500 / tissue_vol.tiv
for i in range(ROIs_BSNIP.shape[0]):
    ROIs_BSNIP.iloc[i,1:] *= tissue_vol.norm_ratio[i]
ROIs_BSNIP.participant_id[0] # 'INV027JRF0P'

# Normalize the BioBD data, using total intracranial volume
ROIs_BIOBD = pd.read_csv('/neurospin/psy_sbox/bipolar/biobd/derivatives/cat12/stats/cat12_rois_Vgm.tsv',sep='\t')
tissue_vol = pd.read_csv('/neurospin/psy_sbox/bipolar/biobd/derivatives/cat12/stats/cat12_tissues_volumes.tsv',sep='\t')
assert ROIs_BIOBD.shape == (746, 143)
tissue_vol['norm_ratio'] = 1500 / tissue_vol.tiv
for i in range(ROIs_BIOBD.shape[0]):
    ROIs_BIOBD.iloc[i,1:] *= tissue_vol.norm_ratio[i]
ROIs_BIOBD.participant_id[0] # 100288468310
type(ROIs_BIOBD.participant_id[0]) # numpy.int64 so need to change to string as the other participants id are strings
ROIs_BIOBD.participant_id = ROIs_BIOBD.participant_id.astype(str)

'848873044557' in list(ROIs_BIOBD.participant_id)

assert list(ROIs_BSNIP) == list(ROIs_BIOBD)
ROIs_BSBD = pd.concat([ROIs_BSNIP, ROIs_BIOBD])
assert list(ROIs_BSBD) == list(ROIs_BSNIP)
assert ROIs_BSBD.shape == (1788, 143)

# to find the data for scz relatives: /neurospin/lnao/Pdiff/josselin/ellen/BSNIP1

# Merge it with phenotype data

pheno_BSBD = pd.read_csv('/neurospin/psy_sbox/start-icaar-eugei/phenotype/BSNIP_BIOBD_Ctrl_Bipolar_201907.csv',sep=',')
pheno_BSBD.subjectID[0] # 'sub-INVFU6KYUU7'
regex = re.compile("sub-([^_]+)")
pheno_BSBD.iloc[:,0] = [regex.findall(s)[0] for s in pheno_BSBD.iloc[:,0]]
df_BSBD = pheno_BSBD[['subjectID','Age','Sex','DX','siteID']]
df_BSBD.columns = ['participant_id','Age','sex_num','DX','siteID']
df_BSBD.sex_num = df_BSBD.sex_num.map({'M':0.0,'F':1.0, 'H':0.0})

dataset_BSBD = pd.merge(df_BSBD, ROIs_BSBD, on='participant_id', how='left')
assert dataset_BSBD.shape == (1028, 147)
# Select the  controls
controls_BSBD = dataset_BSBD[dataset_BSBD.DX == 'HC']
assert controls_BSBD.shape == (570, 147)

# Concatenate the controls from SCZ/PR and BSBD
controls_SC_PR.columns # 'age' 'dx_num' 'sex_num' 'site' , the rest is the same in the two
controls_BSBD.columns # 'Age' 'DX' 'sex_num' 'siteID', ...
controls_BSBD.rename(columns={'Age':'age','siteID':'site','DX':'dx_num'}, inplace=True)
del controls_SC_PR['path']
assert len(controls_SC_PR.columns) == len(controls_BSBD.columns) # 147
list(controls_SC_PR)
# 'participant_id',
# 'dx_num',
# 'sex_num',
# 'site',
# 'age',
list(controls_BSBD)
# 'participant_id',
# 'age',
# 'sex_num',
# 'dx_num',
# 'site',
# reorganize so that pd.concat doesn't reorder
cols = list(controls_BSBD)
cols.index('dx_num') == 3
cols.index('age') == 1
cols.index('participant_id') == 0
cols.index('sex_num') == 2
cols.index('site') == 4
reorder = [cols[0], cols[3], cols[2], cols[4], cols[1]] + [item for item in cols if item not in {'age','dx_num','participant_id','sex_num','site'}]
controls_BSBD = controls_BSBD[reorder]


controls_SC_PR_BSBD = pd.concat([controls_SC_PR, controls_BSBD])
assert controls_SC_PR_BSBD.shape == (990, 147)
assert controls_SC_PR_BSBD[controls_SC_PR_BSBD.isnull().any(axis=1)].shape == (0, 147) # no missing value
for i in list(set(controls_SC_PR_BSBD.site)):
    print(i, len(controls_SC_PR_BSBD[controls_SC_PR_BSBD.site == i]))
#WUSTL 152
#sandiego 74
#udine 90
#galway 41
#Detroit 21
#Baltimore 58
#Boston 26
#creteil 53
#mannheim 38
#vip 53
#grenoble 9
#Dallas 43
#PRAGUE 90
#Hartford 52
#NU 38
#MRN 87
#geneve 28
#pittsburgh 37

# Controls BIOBD = 74 + 38 + 53 + 90 + 41 + 37 + 9 + 28 = 370
    
# Create control dataset without BioBD
    
dataset_BSNIP = pd.merge(df_BSBD, ROIs_BSNIP, on='participant_id', how='left')
assert dataset_BSNIP.shape == (1028, 147)
dataset_BSNIP = dataset_BSNIP.dropna()
assert dataset_BSNIP.shape == (316, 147)
# Select the  controls
controls_BSNIP = dataset_BSNIP[dataset_BSNIP.DX == 'HC']
assert controls_BSNIP.shape == (200, 147)

# Concatenate the controls from SCZ/PR and BSBD
controls_SC_PR.columns # 'age' 'dx_num' 'sex_num' 'site' , the rest is the same in the two
controls_BSNIP.columns # 'Age' 'DX' 'sex_num' 'siteID', ...
controls_BSNIP.rename(columns={'Age':'age','siteID':'site','DX':'dx_num'}, inplace=True)
del controls_SC_PR['path']
assert len(controls_SC_PR.columns) == len(controls_BSNIP.columns) # 147
list(controls_SC_PR)
# 'participant_id',
# 'dx_num',
# 'sex_num',
# 'site',
# 'age',
list(controls_BSNIP)
# 'participant_id',
# 'age',
# 'sex_num',
# 'dx_num',
# 'site',
# reorganize so that pd.concat doesn't reorder
cols = list(controls_BSNIP)
cols.index('dx_num') == 3
cols.index('age') == 1
cols.index('participant_id') == 0
cols.index('sex_num') == 2
cols.index('site') == 4
reorder = [cols[0], cols[3], cols[2], cols[4], cols[1]] + [item for item in cols if item not in {'age','dx_num','participant_id','sex_num','site'}]
controls_BSNIP = controls_BSNIP[reorder]

controls_SC_PR_BSNIP = pd.concat([controls_SC_PR, controls_BSNIP])
assert controls_SC_PR_BSNIP.shape == (620, 147)
assert controls_SC_PR_BSNIP[controls_SC_PR_BSNIP.isnull().any(axis=1)].shape == (0, 147) # no missing value
for i in list(set(controls_SC_PR_BSBD.site)):
    print(i, len(controls_SC_PR_BSBD[controls_SC_PR_BSBD.site == i]))
    
# Create control BIOBD dataset
    
dataset_BIOBD = pd.merge(df_BSBD, ROIs_BIOBD, on='participant_id', how='left')
assert dataset_BIOBD.shape == (1028, 147)
dataset_BIOBD = dataset_BIOBD.dropna()
assert dataset_BIOBD.shape == (710, 147)
# Select the  controls
controls_BIOBD = dataset_BIOBD[dataset_BIOBD.DX == 'HC']
assert controls_BIOBD.shape == (370, 147)


####################### Controls Training set: all controls ########################################

X_all_controls_SC_PR_BS_BD = controls_SC_PR_BSBD.iloc[:,np.r_[5:147]]
assert X_all_controls_SC_PR_BS_BD.shape == (990, 142)
array_X = np.array(X_all_controls_SC_PR_BS_BD)
np.save(os.path.join(OUTPUT_PATH, 'X_all_controls_SCZConn_Pra_BSNIP_BIOBD.npy'), array_X)

y_all_controls_SC_PR_BS_BD = controls_SC_PR_BSBD.age
assert y_all_controls_SC_PR_BS_BD.shape == (990,)
array_y = np.array(y_all_controls_SC_PR_BS_BD)
np.save(os.path.join(OUTPUT_PATH, 'age_all_controls_SCZConn_Pra_BSNIP_BIOBD.npy'), array_y)

sex = controls_SC_PR_BSBD.sex_num
assert sex.shape == (990,) 
array_sex = np.array(sex)
np.save(os.path.join(OUTPUT_PATH, 'sex_all_controls_SCZConn_Pra_BSNIP_BIOBD.npy'), array_sex)

sites = controls_SC_PR_BSBD.site
assert sites.shape == (990,) 
array_sites = np.array(sites)
np.save(os.path.join(OUTPUT_PATH, 'sites_all_controls_SCZConn_Pra_BSNIP_BIOBD.npy'), array_sites)

####################### Controls Training set: without BIOBD ########################################

X_all_controls_SC_PR_BS = controls_SC_PR_BSNIP.iloc[:,np.r_[5:147]]
assert X_all_controls_SC_PR_BS.shape == (620, 142)
array_X = np.array(X_all_controls_SC_PR_BS)
np.save(os.path.join(OUTPUT_PATH, 'X_all_controls_SCZConn_Pra_BSNIP.npy'), array_X)

y_all_controls_SC_PR_BS = controls_SC_PR_BSNIP.age
assert y_all_controls_SC_PR_BS.shape == (620,)
array_y = np.array(y_all_controls_SC_PR_BS)
np.save(os.path.join(OUTPUT_PATH, 'age_all_controls_SCZConn_Pra_BSNIP.npy'), array_y)

sex = controls_SC_PR_BSNIP.sex_num
assert sex.shape == (620,) 
array_sex = np.array(sex)
np.save(os.path.join(OUTPUT_PATH, 'sex_all_controls_SCZConn_Pra_BSNIP.npy'), array_sex)

sites = controls_SC_PR_BSNIP.site
assert sites.shape == (620,) 
array_sites = np.array(sites)
np.save(os.path.join(OUTPUT_PATH, 'sites_all_controls_SCZConn_Pra_BSNIP.npy'), array_sites)

######################## Controls Test set: BIOBD ####################################################

X_controls_BIOBD = controls_BIOBD.iloc[:,np.r_[5:147]]
assert X_controls_BIOBD.shape == (370, 142)
array_X = np.array(X_controls_BIOBD)
np.save(os.path.join(OUTPUT_PATH, 'X_controls_BIOBD.npy'), array_X)

y_controls_BIOBD = controls_BIOBD.Age
assert y_controls_BIOBD.shape == (370,)
array_y = np.array(y_controls_BIOBD)
np.save(os.path.join(OUTPUT_PATH, 'age_controls_BIOBD.npy'), array_y)

sex = controls_BIOBD.sex_num
assert sex.shape == (370,) 
array_sex = np.array(sex)
np.save(os.path.join(OUTPUT_PATH, 'sex_controls_BIOBD.npy'), array_sex)

sites = controls_BIOBD.siteID
assert sites.shape == (370,) 
array_sites = np.array(sites)
np.save(os.path.join(OUTPUT_PATH, 'sites_controls_BIOBD.npy'), array_sites)

#######################################################################################################

# Training set with all controls, START included

# Concatenate SC_PR_BSNIP with START controls = FINAL dataset (for now... waiting for BioBD)
controls_SC_PR_BSNIP_START = pd.concat([controls_SC_PR_BSNIP, start_control])
controls_SC_PR_BSNIP_START.to_csv(os.path.join(OUTPUT_PATH,'norm_dataset_cat12_allcontrols_SCHIZCONNECT_VIP_PRAGUE_BSNIP_START.tsv'), sep='\t')

# Create the X and y sets
X_all_controls = controls_SC_PR_BSNIP_START.iloc[:,np.r_[5:147]]
assert X_all_controls.shape == (370, 142)
array_X = np.array(X_all_controls)
np.save(os.path.join(OUTPUT_PATH, 'X_all_controls.npy'), array_X)

y_age_controls = controls_SC_PR_BSNIP_START.age
assert y_age_controls.shape == (370,)
array_y = np.array(y_age_controls)
np.save(os.path.join(OUTPUT_PATH, 'y_all_controls_age.npy'), array_y)

sites = controls_SC_PR_BSNIP_START.site
assert sites.shape == (370,) 
array_sites = np.array(sites)
np.save(os.path.join(OUTPUT_PATH, 'sites_all_controls.npy'), array_sites)

sex = controls_SC_PR_BSNIP_START.sex_num
assert sex.shape == (370,) 
array_sex = np.array(sex)
np.save(os.path.join(OUTPUT_PATH, 'sex_all_controls.npy'), array_sex)      

"""
Extract the schizophrenia offspring set from BSNIP
"""

###################
# TO BE DONE
###################

#pheno_bsnip = pd.read_csv('/neurospin/psy_sbox/start-icaar-eugei/phenotype/bsnip_all_clinical_data(1).csv',sep=',')
#list(pheno_bsnip)
#pheno_bsnip.phenotype
#pheno_bsnip[(pheno_bsnip.phenotype == 'Relative') & (pheno_bsnip.psychosis_lt == 0)]



