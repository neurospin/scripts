#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:54:23 2019

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

# for the phenotypes
INPUT_CSV_icaar = '/neurospin/psy/start-icaar-eugei/phenotype/raw'
INPUT_CSV_bsnip_biobd = '/neurospin/psy/bipolar/biobd/phenotype/raw'
INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

OUTPUT_PATH = '/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data'


"""
CREATION OF ONE SINGLE DATASET WITH ALL PHENOTYPES
"""

# STEP 4: for each phenotype file, homogeneize the columns

### for ICAAR and START

icaar_eugei = pd.read_csv(os.path.join(INPUT_CSV_icaar,'clinic_icaar_201907.tsv'), sep='\t')
start = pd.read_csv(os.path.join(INPUT_CSV_icaar,'clinic_start_201907.tsv'), sep='\t')

# icaar renommé pour correspondre avec start
icaar_eugei['diagnosis'] = np.nan
for i in range(icaar_eugei.shape[0]):
    if icaar_eugei.Baseline_Status[i] == 'UHR' and icaar_eugei.Conversion[i] == 'yes':
        icaar_eugei.diagnosis[i] = 'UHR-C'
    elif icaar_eugei.Baseline_Status[i] == 'UHR' and icaar_eugei.Conversion[i] == 'no':
        icaar_eugei.diagnosis[i] = 'UHR-NC'
    elif icaar_eugei.Baseline_Status[i] == 'NUHR' and icaar_eugei.Conversion[i] == 'yes':
        icaar_eugei.diagnosis[i] = 'Non-UHR-C'
    elif icaar_eugei.Baseline_Status[i] == 'NUHR' and icaar_eugei.Conversion[i] == 'no':
        icaar_eugei.diagnosis[i] = 'Non-UHR-NC'
    elif icaar_eugei.Baseline_Status[i] == 'Psychotic':
        icaar_eugei.diagnosis[i] = 'Psychotic'
    elif icaar_eugei.Baseline_Status[i] == 'UHR' and pd.isna(icaar_eugei.Conversion[i]):
        icaar_eugei.diagnosis[i] = 'UHR-NaN'
    elif pd.isna(icaar_eugei.Baseline_Status[i]) and pd.isna(icaar_eugei.Conversion[i]):
        icaar_eugei.diagnosis[i] = 'NaN'
icaar_eugei['study'] = 'ICAAR_EUGEI_START'
icaar_eugei['site'] = 'ICM'
icaar_eugei['irm'] = 'M0'
start['study'] = 'ICAAR_EUGEI_START'
start['site'] = 'Sainte-Anne'

# âges de START reformatés
regex = re.compile(".([0-9][0-9]).")
start.iloc[:,3] = [regex.findall(s)[0] for s in start.iloc[:,3]]
for i in range(start.shape[0]):
    start.age.iloc[i] = float(start.age[i])
    
start = start.rename(columns = {'clinical_status':'diagnosis'})
start['alcohol_last_month'] = np.nan


df_icaar = icaar_eugei[['participant_id','age','sex','diagnosis', 'study','site', 'medication', 
                        'cannabis_last_month', 'tobacco_last_month', 'alcohol_last_month',
                        'BPRS', 'PANSS_total', 'PANSS_positive', 'PANSS_negative',
                        'PANSS_psychopatho', 'PANSS_desorganisation', 'SANS', 'SAPS', 'MADRS',
                        'SOFAS', 'NSS', 'irm']]


df_start = start[['participant_id','age','sex','diagnosis','study','site','medication',
                  'cannabis_last_month', 'tobacco_last_month', 'alcohol_last_month', 
                  'BPRS', 'PANSS_total', 'PANSS_positive', 'PANSS_negative', 
                  'PANSS_psychopatho', 'PANSS_desorganisation', 'SANS', 'SAPS', 'MADRS', 'SOFAS', 'NSS','irm']]

icaar_eugei_start = pd.concat([df_icaar, df_start])
assert icaar_eugei_start.shape == (170, 22)
icaar_eugei_start.sex = icaar_eugei_start.sex.map({'F':1.0,'H':0.0,'M':0.0})
 
pheno_icaar_eugei_start = icaar_eugei_start.reindex(columns = icaar_eugei_start.columns.tolist() + ['Age of Onset',
 'Alcohol',
 'Anticonvulsants',
 'Antidepressants',
 'Antipsychotics',
 'BD Type',
 'Density of Episodes',
 'Depression Scale',
 'Depression Score',
 'Illness Duration',
 'Lithium',
 'Mania Scale',
 'Mania Score',
 'Mood Phase',
 'Number of Depressive Episodes',
 'Number of Manic Episodes',
 'Onset Time',
 'Psychotic',
 'Severity',
 'Total Episodes', 'ymrstot','psysoc_65','psychosis_lt','phenotype'])

assert pheno_icaar_eugei_start.shape == (170, 46)
assert pheno_icaar_eugei_start.duplicated().sum() == 0


### for SchizConnect, VIP

phenotype_SC = pd.read_csv(INPUT_CSV_schizconnect, sep='\t')
list(phenotype_SC) # ['participant_id', 'dx_num', 'path', 'sex_num', 'site', 'age']
phenotype_SC['study'] = 'SCHIZCONNECT-VIP'
phenotype_SC['irm'] = np.nan
phenotype_SC = phenotype_SC.rename(columns = {'dx_num':'diagnosis','sex_num':'sex'})
phenotype_SC.diagnosis = phenotype_SC.diagnosis.map({1.0:'chronic schizophrenia',0.0:'control'})

pheno_SC = phenotype_SC[['participant_id','age','sex','diagnosis','study','site','irm']]
assert pheno_SC.shape == (606, 7)

pheno_SC = pheno_SC.reindex(columns = pheno_SC.columns.tolist() + ['medication','Age of Onset',
 'Alcohol',
 'Anticonvulsants',
 'Antidepressants',
 'Antipsychotics',
 'BD Type',
 'Density of Episodes',
 'Depression Scale',
 'Depression Score',
 'Illness Duration',
 'Lithium',
 'Mania Scale',
 'Mania Score',
 'Mood Phase',
 'Number of Depressive Episodes',
 'Number of Manic Episodes',
 'Onset Time',
 'Psychotic',
 'Severity',
 'Total Episodes',
 'cannabis_last_month',
 'tobacco_last_month',
 'alcohol_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'MADRS',
 'SOFAS',
 'NSS', 'ymrstot','psychosis_lt', 'psysoc_65','phenotype'])

assert pheno_SC.shape == (606, 46)
assert pheno_SC.duplicated().sum() == 0


### for PRAGUE

phenotype_PR = pd.read_csv(INPUT_CSV_prague, sep='\t')
list(phenotype_PR) # ['participant_id', 'dx_num', 'sex_num', 'site', 'age', 'path']
phenotype_PR['study'] = 'PRAGUE'
phenotype_PR['irm'] = np.nan
phenotype_PR = phenotype_PR.rename(columns = {'dx_num':'diagnosis','sex_num':'sex'})
phenotype_PR.diagnosis = phenotype_PR.diagnosis.map({1.0:'FEP',0.0:'control'})

pheno_PR = phenotype_PR[['participant_id','age','sex','diagnosis','study','site','irm']]
assert pheno_PR.shape == (133, 7)
assert pheno_PR.duplicated().sum() == 0

pheno_PR = pheno_PR.reindex(columns = pheno_PR.columns.tolist() + ['medication','Age of Onset',
 'Alcohol',
 'Anticonvulsants',
 'Antidepressants',
 'Antipsychotics',
 'BD Type',
 'Density of Episodes',
 'Depression Scale',
 'Depression Score',
 'Illness Duration',
 'Lithium',
 'Mania Scale',
 'Mania Score',
 'Mood Phase',
 'Number of Depressive Episodes',
 'Number of Manic Episodes',
 'Onset Time',
 'Psychotic',
 'Severity',
 'Total Episodes',
 'cannabis_last_month',
 'tobacco_last_month',
 'alcohol_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'MADRS',
 'SOFAS',
 'NSS', 'ymrstot','psychosis_lt','psysoc_65','phenotype'])

assert pheno_PR.shape == (133, 46)
assert pheno_PR.duplicated().sum() == 0


### for BSNIP and BIOBD

pheno_BSBD = pd.read_csv(os.path.join(INPUT_CSV_bsnip_biobd, 'BSNIP_BIOBD_Ctrl_Bipolar_201907.csv'),sep=',')
# healthy controls and bipolars from BSNIP and BIOBD
pheno_BSBD.subjectID[0] # 'sub-INVFU6KYUU7'
regex = re.compile("sub-([^_]+)")
pheno_BSBD.iloc[:,0] = [regex.findall(s)[0] for s in pheno_BSBD.iloc[:,0]]
pheno_BSBD = pheno_BSBD.rename(columns = {'subjectID':'participant_id','siteID':'site','Age':'age','Sex':'sex','DX':'diagnosis','On Meds':'medication'})
pheno_BSBD['study'] = np.nan
pheno_BSBD['irm'] = np.nan 
for i in range(pheno_BSBD.shape[0]):
    try:
        int(pheno_BSBD.participant_id[i])
        pheno_BSBD.study[i] = 'BIOBD'
    except:
        pheno_BSBD.study[i] = 'BSNIP'
pheno_BSBD = pheno_BSBD.reindex(columns = pheno_BSBD.columns.tolist() + ['cannabis_last_month',
 'tobacco_last_month',
 'alcohol_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'MADRS',
 'SOFAS',
 'NSS', 'ymrstot','psychosis_lt','psysoc_65','phenotype']) 
assert pheno_BSBD.shape == (1028, 46)
assert pheno_BSBD.duplicated().sum() == 0

pheno_BIOBD = pheno_BSBD[pheno_BSBD.study == 'BIOBD']
assert pheno_BIOBD.shape == (711, 46)

pheno_BSNIP_1 = pheno_BSBD[pheno_BSBD.study == 'BSNIP']
assert pheno_BSNIP_1.shape == (317, 46)


pheno_BSNIP_2 = pd.read_csv('/neurospin/psy_sbox/start-icaar-eugei/phenotype/BSNIP_all_clinical_data.csv',sep=',')
assert pheno_BSNIP_2.shape == (1094, 232)
assert pheno_BSNIP_2.duplicated().sum() == 0
pheno_relatives = pd.read_csv('/neurospin/psy_sbox/start-icaar-eugei/phenotype/BSNIP_cases_ctrls_relatives_diagnosis.csv')
assert pheno_relatives.shape == (4125, 2)
assert pheno_relatives.duplicated().sum() == 1874
pheno_relatives = pheno_relatives.drop_duplicates()
assert pheno_relatives.shape == (2251, 2)
pheno_BSNIP_2 = pd.merge(pheno_BSNIP_2, pheno_relatives, on='subjectkey', how='outer')
assert pheno_BSNIP_2.shape == (2251, 233)

pheno_BSNIP_2 = pheno_BSNIP_2[['subjectkey','interview_age_x','gender_x','site_x','madrstot','ymrstot','diagnosis','psychosis_lt','psysoc_65','phenotype']]
assert pheno_BSNIP_2.diagnosis.isnull().sum() == 0
pheno_BSNIP_2 = pheno_BSNIP_2.rename(columns = {'subjectkey':'participant_id','site_x':'site','interview_age_x':'age','gender_x':'sex','madrstot':'MADRS','psychosis_lt':'psychosis_lt','phenotype':'phenotype','psysoc_65':'psysoc_65'})
regex = re.compile("NDAR_([^_]+)")
pheno_BSNIP_2.iloc[:,0] = [regex.findall(s)[0] for s in pheno_BSNIP_2.iloc[:,0]]
pheno_BSNIP_2['study'] = 'BSNIP'
pheno_BSNIP_2['irm'] = np.nan
for i in range(pheno_BSNIP_2.shape[0]):
    pheno_BSNIP_2.age[i] *= 1/12 
pheno_BSNIP_2 = pheno_BSNIP_2.reindex(columns = pheno_BSNIP_2.columns.tolist() + ['medication','Age of Onset',
 'Alcohol',
 'Anticonvulsants',
 'Antidepressants',
 'Antipsychotics',
 'BD Type',
 'Density of Episodes',
 'Depression Scale',
 'Depression Score',
 'Illness Duration',
 'Lithium',
 'Mania Scale',
 'Mania Score',
 'Mood Phase',
 'Number of Depressive Episodes',
 'Number of Manic Episodes',
 'Onset Time',
 'Psychotic',
 'Severity',
 'Total Episodes',
 'cannabis_last_month',
 'tobacco_last_month',
 'alcohol_last_month',
 'BPRS',
 'PANSS_total',
 'PANSS_positive',
 'PANSS_negative',
 'PANSS_psychopatho',
 'PANSS_desorganisation',
 'SANS',
 'SAPS',
 'SOFAS',
 'NSS'])
assert pheno_BSNIP_2.shape == (2251, 46)

# the subjects from pheno_BSNIP_1 are all in pheno_BSNIP_2
for i in list(pheno_BSNIP_1.participant_id):
    if i not in list(pheno_BSNIP_2.participant_id):
        print(i)

# donc:
pheno_BSNIP = pheno_BSNIP_2


# STEP 5: concatenate all

# check all have same columns
assert set(list(pheno_icaar_eugei_start)) == set(list(pheno_SC))
assert set(list(pheno_icaar_eugei_start)) == set(list(pheno_PR))
assert set(list(pheno_icaar_eugei_start)) == set(list(pheno_BIOBD))
assert set(list(pheno_icaar_eugei_start)) == set(list(pheno_BSNIP))

pheno_all = pd.concat([pheno_icaar_eugei_start, pheno_SC, pheno_PR, pheno_BIOBD, pheno_BSNIP])
assert pheno_all.shape == (3871, 46)
assert pheno_all.duplicated('participant_id').sum() == 0


# reorder columns

list_col = list(pheno_all)
assert list_col.index('participant_id') == 37
assert list_col.index('sex') == 41
assert list_col.index('age') == 31
assert list_col.index('diagnosis') == 34
assert list_col.index('study') == 43
assert list_col.index('site') == 42
reorder_col = [list_col[37], list_col[41], list_col[31], list_col[34], list_col[43], list_col[42]] + [item for item in list_col if item not in {'participant_id','sex','age','diagnosis','study','site'}]
pheno_all = pheno_all[reorder_col]
assert pheno_all.shape == (3871, 46)


# homogeneize age coding to float

for i in range(pheno_all.shape[0]):
    if isinstance(pheno_all.age.iloc[i], str):
        pheno_all.age.iloc[i] = pheno_all.age.iloc[i].replace(',','.')
        pheno_all.age.iloc[i] = float(pheno_all.age.iloc[i])
    elif isinstance(pheno_all.age.iloc[i], int):
        pheno_all.age.iloc[i] = float(pheno_all.age.iloc[i])

# homogeneize diagnosis coding

pheno_all.diagnosis = pheno_all.diagnosis.map({
'UHR-C':'UHR-C',
'UHR-NC':'UHR-NC',
'UHR-NaN':'UHR-NaN',
'Non-UHR-NC':'Non-UHR-NC',
'Psychotic':'Psychotic',
'Non-UHR':'Non-UHR',
'Retard_Mental':'Retard_Mental',
'control':'control',
'FEP':'FEP',   
'chronic schizophrenia':'schizophrenia',
'SCZ':'schizophrenia',
'CTRL':'control',
'BIP-III':'bipolar disorder',
'HC':'control',
'BD':'bipolar disorder',
'ADHD, SU':'ADHD, SU',
'EDM':'EDM',
'MDE, ADHD, panic':'MDE, ADHD, panic',
'SU, panic':'SU, panic',
'MDE, PTSD':'MDE, PTSD',
'ADHD':'ADHD',
'Healthy Control':'control',
'Proband with Schizophrenia':'schizophrenia',
'Proband with Schizoaffective Disorder':'schizoaffective disorder',
'Proband with Psychotic Bipolar Disorder':'psychotic bipolar disorder',
'Relative of Proband with Schizophrenia':'relative of proband with schizophrenia',
'Relative of Proband with Schizoaffective Disorder':'relative of proband with schizoaffective disorder',
'Relative of Proband with Psychotic Bipolar Disorder':'relative of proband with psychotic bipolar disorder'})

# homogeneize sex coding
    
pheno_all.sex = pheno_all.sex.map({'F':1.0,'H':0.0,'M':0.0, 1.0:1.0, 0.0:0.0})     
        


pheno_all.to_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START.tsv'), sep='\t', index=False)


##### Make ABIDE 2 phenotype

ABIDE2_BASEPATH = '/neurospin/psy/abide2'
OUTPUT_PATH_ABIDE2 = '/neurospin/psy_sbox/hc/abide2/participants.tsv'
all_pheno_filenames = glob.glob(os.path.join(ABIDE2_BASEPATH, 'raw/*.csv'))
basic_pheno_filename = os.path.join(ABIDE2_BASEPATH, 'derivatives/pynet/participants.tsv')

# Purpose: align the basic demographic info contained in "all_pheno" with the order defined in "basic_pheno".
# This will be used by the PyNet library.

output_pheno = pd.read_csv(basic_pheno_filename, sep='\t')
# Concatenates all the complete phenotypes
all_pheno_complete = []
for file in all_pheno_filenames:
    if file != '/neurospin/psy/abide2/raw/ABIDEII_Composite_Phenotypic.csv':
        try:
            pheno_complete = pd.read_csv(file, sep=',', encoding='ISO-8859-1')
            pheno_complete = pheno_complete.rename(columns={'SUB_ID': 'participant_id'})
            all_pheno_complete.append(pheno_complete)
        except Exception as e:
            print(e, file)


all_pheno_complete = pd.concat(all_pheno_complete, ignore_index=True, sort=False)
assert all_pheno_complete.shape == (1114, 350)
assert len(set(all_pheno_complete.participant_id)) == len(all_pheno_complete)

# Basic checks
assert set(all_pheno_complete['participant_id']) >= set(output_pheno['participant_id'])
sorted_p_id = list(output_pheno.participant_id)
len_out_pheno = len(output_pheno)

# Merge the 2 dataframes
output_pheno = pd.merge(output_pheno, all_pheno_complete, on='participant_id', how='left', sort=False)
assert len_out_pheno == len(output_pheno)
assert sorted_p_id == list(output_pheno.participant_id)

output_pheno = output_pheno.rename(columns={'SEX': 'sex', 'AGE_AT_SCAN ': 'age', 'DX_GROUP': 'diagnosis', 'center': 'site'})
output_pheno.diagnosis = output_pheno.diagnosis.map({1: 'autism', 2: 'control'})
output_pheno.sex = output_pheno.sex.map({2: 1, 1: 0}) # Male == 0, Female == 1
output_pheno['study'] = 'ABIDE2'
output_pheno.to_csv(OUTPUT_PATH_ABIDE2, sep='\t', index=False)

##### Make IXI phenotype and merge ABIDE2 + IXI
IXI_PATH = '/neurospin/psy_sbox/hc/ixi/'
OUTPUT_PATH = '/neurospin/psy_sbox/hc/abide2_ixi_participants.tsv'
pheno_ixi = pd.read_csv(os.path.join(IXI_PATH, 'participants.tsv'), sep='\t')

pheno_ixi.sex = pheno_ixi.sex.map({'M': 0, 'F': 1})
pheno_ixi['study'] = 'IXI'
pheno_ixi['site'] = 'LONDON'
pheno_ixi['diagnosis'] = 'control'
participant_ids_ixi, participant_ids_abide2 = pheno_ixi.participant_id.values, output_pheno.participant_id.values
pheno_abide2_ixi = pd.concat([output_pheno, pheno_ixi], ignore_index=True, sort=False)
assert list(pheno_abide2_ixi.participant_id.values) == list(participant_ids_abide2) + list(participant_ids_ixi)
assert pheno_ixi.shape == (581, 9)
assert output_pheno.shape == (1108, 352)
assert pheno_abide2_ixi.shape == (581+1108, 9+352-6) #6 common columns: age, sex, diagnosis, study, site, participant_id

pheno_abide2_ixi.to_csv(OUTPUT_PATH, sep='\t', index=False)


## Make IXI phenotype with TIV
IXI_PATH = '/neurospin/psy_sbox/hc/ixi/'
OUTPUT_PATH = '/neurospin/psy_sbox/hc/ixi/IXI_t1mri_mwp1_participants.csv'
pheno_ixi = pd.read_csv(os.path.join(IXI_PATH, 'participants.tsv'), sep='\t')
meta_data_ixi = pd.read_csv(os.path.join(IXI_PATH, 'tiv.csv'), sep=',')
pheno_ixi.sex = pheno_ixi.sex.map({'M': 0, 'F': 1})
pheno_ixi['study'] = 'IXI'
pheno_ixi['site'] = 'LONDON'
pheno_ixi['diagnosis'] = 'control'
meta_data_ixi = meta_data_ixi.rename(columns={'TIV': 'tiv'})
pheno_ixi = pd.merge(pheno_ixi, meta_data_ixi, on='participant_id', how='left', sort=False)

pheno_ixi.to_csv(OUTPUT_PATH, sep='\t', index=False)

## Make ABIDE2 phenotype with TIV
ABIDE2_PATH = '/neurospin/psy_sbox/hc/abide2/'
OUTPUT_PATH = '/neurospin/psy_sbox/hc/abide2/ABIDE2_t1mri_mwp1_participants.csv'
pheno_abide2 = pd.read_csv(os.path.join(ABIDE2_PATH, 'participants.tsv'), sep='\t')
meta_data_abide2 = pd.read_csv(os.path.join(ABIDE2_PATH, 'derivatives/cat12_estimations.csv'), sep=',')
meta_data_abide2 = meta_data_abide2.rename(columns={'TIV': 'tiv'})
pheno_abide2 = pd.merge(pheno_abide2, meta_data_abide2, on='participant_id', how='left', sort=False)
pheno_abide2 = pheno_abide2[~pheno_abide2.tiv.isna()]

pheno_abide2.to_csv(OUTPUT_PATH, sep='\t', index=False)







