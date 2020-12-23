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

###############################################################################
# Make phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20190909.tsv

# for the phenotypes
INPUT_CSV_icaar = '/neurospin/psy/start-icaar-eugei/phenotype/raw'
INPUT_CSV_bsnip_biobd = '/neurospin/psy/biobd/phenotype/raw'
INPUT_CSV_schizconnect = '/neurospin/psy/schizconnect-vip-prague/participants_schizconnect-vip.tsv'
INPUT_CSV_prague = '/neurospin/psy/schizconnect-vip-prague/participants_prague.tsv'

OUTPUT_PATH = '/neurospin/psy/all_studies/phenotype'


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


pheno_BSNIP_2 = pd.read_csv(os.path.join(INPUT_CSV_bsnip_biobd, 'BSNIP_all_clinical_data.csv'),sep=',')
assert pheno_BSNIP_2.shape == (1094, 232)
assert pheno_BSNIP_2.duplicated().sum() == 0
pheno_relatives = pd.read_csv(os.path.join(INPUT_CSV_bsnip_biobd, 'BSNIP_cases_ctrls_relatives_diagnosis.csv'))
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
head_cols = ['participant_id', 'sex', 'age', 'diagnosis', 'study', 'site']
list_cols = list(pheno_all)
reorder_col = head_cols + [item for item in list_cols if item not in head_cols]
# assert list_col.index('participant_id') == 37
# assert list_col.index('sex') == 41
# assert list_col.index('age') == 31
# assert list_col.index('diagnosis') == 34
# assert list_col.index('study') == 43
# assert list_col.index('site') == 42
# reorder_col = [list_col[37], list_col[41], list_col[31], list_col[34], list_col[43], list_col[42]] + [item for item in list_col if item not in {'participant_id','sex','age','diagnosis','study','site'}]
pheno_all = pheno_all[reorder_col]
assert pheno_all.shape == (3871, 46)


# homogeneize age coding to float

for i in range(pheno_all.shape[0]):
    if isinstance(pheno_all.age.iloc[i], str):
        pheno_all.age.iloc[i] = pheno_all.age.iloc[i].replace(',','.')
        pheno_all.age.iloc[i] = float(pheno_all.age.iloc[i])
    elif isinstance(pheno_all.age.iloc[i], int):
        pheno_all.age.iloc[i] = float(pheno_all.age.iloc[i])

pheno_all.age = pheno_all.age.astype(float)

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


# homogeneize tobacco_last_month

mapping = {lev:lev for lev in pheno_all.tobacco_last_month.unique()}
mapping['ND'] = np.nan
mapping = {k:float(v) for k, v in mapping.items()}

pheno_all.tobacco_last_month = pheno_all.tobacco_last_month.map(mapping)


# Drop index

pheno_all = pheno_all.reset_index(drop=True)

pheno_all.to_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20190909.tsv'), sep='\t', index=False)

# Check differences with other version
def _compare_df(df1, df2):
    df2 = df2[df1.columns]

    # Ensure tocaco coding
    mapping = {lev:lev for lev in df2.tobacco_last_month.unique()}
    mapping['ND'] = np.nan
    mapping = {k:float(v) for k, v in mapping.items()}
    df2.tobacco_last_month = df2.tobacco_last_month.map(mapping)

    # Round age at 10-6
    df1.age = df1.age.round(decimals=6)
    df2.age = df2.age.round(decimals=6)

    # Round Density of Episodes at 10-6
    df1['Density of Episodes'] = df1['Density of Episodes'].round(decimals=6)
    df2['Density of Episodes'] = df2['Density of Episodes'].round(decimals=6)

    for i in range(df1.shape[0]):
        if not df1.iloc[i].equals(df2.iloc[i]):
            row1 = df1.iloc[i].to_frame().T
            row2 = df2.iloc[i].to_frame().T
            diff = pd.concat([row1, row2]).drop_duplicates(keep=False)
            if len(diff) > 0:
                print(diff)
                for n in row1.columns:
                    val1, val2 = row1[n].values[0], row2[n].values[0]
                    if (pd.isnull(val1) and pd.isnull(val2)):
                        continue
                    elif val1 != val2:
                        print(row1.participant_id, n, val1, val2)
                        raise "Differences found"

    assert df1.equals(df2), "Global differences found"


new = pd.read_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20190909.tsv'), sep='\t')
old = pd.read_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20190909_orig.tsv'), sep='\t')

_compare_df(new, old)


###############################################################################
# Intergrate modification by Anton on ICAAR-START
# iftimovici anton <anton.iftimovici@gmail.com>
# to:	Edouard Duchesnay <duchesnay@gmail.com>
# date:	May 14, 2020, 9:41 AM

# 1) J'ai fait une re-vérification globale de tous nos phénotypes ICAAR-START, y compris ceux des patients qui ont de l'imagerie:
# - une erreur sur l'âge d'un UHR-non converteur a été retrouvée (il passe de 23 à 18 ans)
# - plusieurs sujets n'étaient pas inclus dans nos études car ils étaient Non-UHR, et je n'avais pris que des UHR initialement; sauf que si on compare converteurs vs non-converteurs, ce n'est pas un problème; ça rajoute 4 IRM de non-converteurs à M0 et MF.
# - j'ai calculé des âges précis qui correspondent à trois temps différents, celui de la passation de l'irm, celui de la passation des tests neuropsycho, et celui des prélèvements biologiques (la différence entre eux n'est pas significative, mais ça m'a surtout servi pour compléter les âges des sujets dont on ne connaissait pas l'âge au moment de l'irm

# 2) J'ai récupéré tous les phénotypes de cognition pour ICAAR-START, que j'ai homogénéisé et ensuite standardisé en fonctions des moyennes et déviations standard pour chaque note dans la population générale, en tenant compte des critères spécifiques de chaque test, d'après les indications de Célia (années d'étude et sexe, en général); du coup, ce sont des données de cognition vraiment très propres, qu'on pourra utiliser !

df = pd.read_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20190909.tsv'), sep='\t')
anton = pd.read_csv(os.path.join(OUTPUT_PATH,'sources/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20200626_from_anton.tsv'), sep='\t')
# Found 21 year in in Anton file
df.loc[df.participant_id == 'STARTLF170705', 'age'] = 21
df.loc[df.participant_id == 'STARTHW170703', 'diagnosis'] = 'UHR-NC'
df.loc[df.participant_id == 'STARTNA150400', 'diagnosis'] = 'UHR-NC'
df.loc[df.participant_id == 'STARTNA160597', 'diagnosis'] = 'UHR-NC'
df.loc[df.participant_id == 'STARTLF160491', 'age'] = 20

_compare_df(df1=df, df2=anton)

df.to_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20200626.tsv'), sep='\t', index=False)

###############################################################################
    tivo_schizconnect = pd.read_csv(os.path.join(STUDY_PATH_schizconnect,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]

    # tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    tivo_biobd = pd.read_csv(os.path.join(STUDY_PATH,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)

    # assert tivo_icaar.shape == (171, 6)
    # assert len(ni_icaar_filenames) == 171

    # assert tivo_schizconnect.shape == (738, 6)
    # assert len(ni_schizconnect_filenames) == 738

    # assert tivo_bsnip.shape == (1042, 6)
    # assert len(ni_bsnip_filenames) == 1042

    assert tivo_biobd.shape ==  (746, 5)


###############################################################################
# QC with Laurie Anne

df = pd.read_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20200626.tsv'), sep='\t')
df.shape == (3871, 46)

other = df[~df.study.isin(['BIOBD', 'BSNIP'])]

df = df[df.study.isin(['BIOBD', 'BSNIP'])]
df.shape == (2962, 46)

LAURIEANNE_CSV = "/neurospin/psy_sbox/bipolar-biobd/phenotype/2019_laurie_anne/biobd_bsnip_clinical.csv"

laurie = pd.read_csv(LAURIEANNE_CSV, sep=',')
laurie.shape == (1099, 26)
laurie.insert(0, 'participant_id', laurie.subjectID.str.replace("\.0", ""))

# Fix Age
laurie['Age'] = laurie['Age'].apply(lambda x: float(x.replace(',','.')) if pd.notnull(x) else None)
laurie.Sex = laurie.Sex.map({'M':'M', 'H':'M', 'F':'F'})
mapping = {v:v for v in laurie.DX.unique()}
mapping['HC'] = 'control'
mapping['BD'] = 'bipolar disorder'
mapping['SZ'] = 'schizophrenia'
mapping['0'] = np.nan
laurie.DX = laurie.DX.map(mapping)

# Some participants are controls from
# "/neurospin/psy_sbox/bipolar-biobd/derivatives/cat12-12.6_vbm_qc-laurie-anne/norm_dataset_cat12_bsnip_biobd.tsv"
ctrl = ['802708704329', '127567692450', '755449777673', '344463774916', '184173668405', '206152431141', '876651563587', '460127580829', '298882272395', '799399693953', '664703775284', '994959841722', '866203430929', '911350678973', '261904584888', '143031618176', '331968608917', '714476111882', '188862939450', '464576291986', '639944271186', '718400641243', '386747285357', '939887150837', '373702396205', '558753042196', '154558208022', '403707311299']
laurie.loc[laurie.participant_id.isin(ctrl), "DX"] = 'control'

# QC between Participants and Laurie file
# recode 'psychotic bipolar disorder' => 'bipolar disorder' before comparison
mapping = {v:v for v in df.diagnosis.unique()}
mapping['psychotic bipolar disorder'] = 'bipolar disorder'
df.diagnosis = df.diagnosis.map(mapping)

merge_ = pd.merge(
        df[['participant_id',  'site', 'sex', 'age', 'diagnosis', 'study', 'Age of Onset']],
        laurie[['participant_id', 'siteID', 'Sex', 'Age', 'DX', 'Age of Onset']],
        on='participant_id', suffixes=('_anton', '_laurie'))
assert merge_.shape[0] == 711  # 711 subjects

# Age sex site
assert np.all(merge_.site == merge_.siteID)
assert np.all(merge_.sex == merge_.Sex.map({'M':0, 'H':0, 'F':1}))
assert np.all((merge_.age - merge_.Age) < 1e-3)

# DX
diff_ = merge_[~merge_.diagnosis.eq(merge_.DX)]# & (merge_.diagnosis.notnull() & diff_.DX.notnull())]
diff_ = diff_[np.logical_not(diff_.diagnosis.isnull() & diff_.DX.isnull())]
assert len(diff_) == 1

print("Differ for only on subject")
print(diff_)

# Age of Onset
assert np.allclose(merge_["Age of Onset_anton"],  merge_['Age of Onset_laurie'], equal_nan=True)

###############################################################################
#%% FIX Remove subjects from biobd subject dublicated in schizconnect(vip)

if False:
    # Match subjects using Total Imaging volumes
    STUDY_PATH_biodb = '/neurospin/psy_sbox/bipolar-biobd'
    STUDY_PATH_schizconnect = '/neurospin/psy/schizconnect-vip-prague'

    # Read schizconnect to remove duplicates
    # tivo_icaar = pd.read_csv(os.path.join(BASE_PATH_icaar, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    #tivo_schizconnect = pd.read_csv(os.path.join(STUDY_PATH_schizconnect, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    vol_cols = ["participant_id", 'TIV', 'CSF_Vol', 'GM_Vol', 'WM_Vol']

    tivo_schizconnect = pd.read_csv(os.path.join(STUDY_PATH_schizconnect,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    assert tivo_schizconnect.shape ==  (738, 5)

    # tivo_bsnip = pd.read_csv(os.path.join(BASE_PATH_bsnip, 'stats', 'cat12_tissues_volumes.tsv'), sep='\t')
    tivo_biobd = pd.read_csv(os.path.join(STUDY_PATH_biodb,
        'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')[vol_cols]
    tivo_biobd.participant_id = tivo_biobd.participant_id.astype(str)
    assert tivo_biobd.shape ==  (746, 5)


    df = tivo_biobd.append(tivo_schizconnect)
    duplicated_in_biobd =  df["participant_id"][df.loc[:, vol_cols[1:]].duplicated(keep='last')]
    assert len(duplicated_in_biobd) == 14

duplicated_in_biobd = ['341879365063', '156634941156', '611954003219', '999412570656', '435432648506', '186334059458', '870810930661', '153138320244', '726278928908', '611553851411', '942465208526', '148210353882', '419555247213', '544435731463']


df = pd.read_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20200626.tsv'), sep='\t')
df.shape == (3871, 46)

assert np.all(df.loc[df.participant_id.isin(duplicated_in_biobd), "study"] == "BIOBD")
df = df.loc[~df.participant_id.isin(duplicated_in_biobd)]
df.shape == (3857, 46)

df.to_csv(os.path.join(OUTPUT_PATH,'phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_20201223.tsv'), sep='\t', index=False)


###############################################################################
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


## Make HCP phenotype
HCP_PATH = '/neurospin/psy_sbox/hcp'
sex_dx_site = pd.read_csv(os.path.join(HCP_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(HCP_PATH, 'hcp_restricted_data.csv'), sep=',')
age = age.rename(columns={'Subject': 'participant_id', 'Age_in_Yrs': 'age'})
sex_dx_site.drop(columns='age', inplace=True)
age_sex_dx_site_study_tiv = pd.merge(sex_dx_site, age, on="participant_id", how='left', validate='1:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.tiv.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna() &
                                                      ~age_sex_dx_site_study_tiv.site.isna()
                                                      ]
assert len(age_sex_dx_site_study_tiv) == 1113 and age_sex_dx_site_study_tiv.participant_id.is_unique
age_sex_dx_site_study_tiv.to_csv(os.path.join(HCP_PATH, 'HCP_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make IXI phenotype with TIV
IXI_PATH = '/neurospin/psy_sbox/ixi/'
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

## Make ABIDE1 phenotype
## Dataset dependent
ABIDE1_PATH = '/neurospin/psy_sbox/abide1/'
age_sex_dx_site = pd.read_csv(os.path.join(ABIDE1_PATH, 'Phenotypic_V1_0b.csv'), sep=',')
age_sex_dx_site = age_sex_dx_site.rename(columns={"AGE_AT_SCAN": 'age', 'SEX': 'sex', 'SITE_ID': 'site',
                                                  "DX_GROUP": "diagnosis", "SUB_ID": 'participant_id'})
age_sex_dx_site.diagnosis = age_sex_dx_site.diagnosis.map({1: 'autism', 2:'control'})
age_sex_dx_site.participant_id = age_sex_dx_site.participant_id.astype(str)
age_sex_dx_site.sex = age_sex_dx_site.sex.map({1:0, 2:1}) # 1: Male, 2: Female
age_sex_dx_site['study'] = 'ABIDE1'
assert age_sex_dx_site.participant_id.is_unique

## Dataset-independent
tiv = pd.read_csv(os.path.join(ABIDE1_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna() &
                                                      ~age_sex_dx_site_study_tiv.site.isna()
                                                      ]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 1 # No TIV available for participant 50818
age_sex_dx_site_study_tiv.to_csv(os.path.join(ABIDE1_PATH, 'ABIDE1_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make ABIDE2 phenotype with TIV
ABIDE2_PATH = '/neurospin/psy_sbox/abide2/'
OUTPUT_PATH = '/neurospin/psy_sbox/abide2/ABIDE2_t1mri_mwp1_participants.csv'
pheno_abide2 = pd.read_csv(os.path.join(ABIDE2_PATH, 'participants.tsv'), sep='\t')
pheno_abide2.participant_id = pheno_abide2.participant_id.astype(str)
assert pheno_abide2.participant_id.is_unique
## Dataset-independent
tiv = pd.read_csv(os.path.join(ABIDE2_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, pheno_abide2, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 11
age_sex_dx_site_study_tiv.to_csv(os.path.join(ABIDE2_PATH, 'ABIDE2_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make CoRR phenotype with TIV
CORR_PATH = '/neurospin/psy_sbox/CoRR'
## Dataset-dependent
sex_site = pd.read_csv(os.path.join(CORR_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(CORR_PATH, 'MR_sessions.tsv'), sep='\t')
filter_age = age['MR ID'].str.contains('baseline')
age = age[filter_age]
## TODO: merge the data to have: <participant_id> <age> <sex> <diagnosis> <site> <study> <...>
sex_site = sex_site.rename(columns={'M/F': 'sex'})
sex_site.sex = sex_site.sex.map({'M':0, 'F':1})
sex_site = sex_site[~sex_site.sex.isna()] # Erases 7 participants
sex_site['site'] = sex_site['Subject'].str.extract(r'(\w+)_([0-9]+)', expand=True)[0]
sex_site['participant_id'] = sex_site['Subject'].str.extract(r'(\w+)_([0-9]+)', expand=True)[1]
assert len(sex_site.participant_id) == len(set(sex_site.participant_id)) == 1379 # Unique participant_id
age_sex_dx_site_study = pd.merge(sex_site, age, on='Subject', how='left', sort=False, validate='1:1')
age_sex_dx_site_study = age_sex_dx_site_study.rename(columns={'Age': 'age'})
age_sex_dx_site_study['study'] = 'CoRR'
age_sex_dx_site_study['diagnosis'] = 'control'
assert np.all(~age_sex_dx_site_study.age.isna() & ~age_sex_dx_site_study.sex.isna())
assert len(age_sex_dx_site_study.participant_id) == len(set(age_sex_dx_site_study.participant_id)) == 1379
## Dataset-independent
tiv = pd.read_csv(os.path.join(CORR_PATH, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge everything
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site_study, on='participant_id', how='left', sort=False,
                                     validate='m:1')
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 2698
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 13
age_sex_dx_site_study_tiv.to_csv(os.path.join(CORR_PATH, 'CoRR_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make NAR phenotype
NAR_path = '/neurospin/psy_sbox/nar'
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(NAR_path, 'participants.tsv'), sep='\t')
assert len(age_sex_dx) == 315
age_sex_dx.sex =  age_sex_dx.sex.str.split(',', expand=True)[0]
age_sex_dx.sex = age_sex_dx.sex.map({'M':0, 'F':1})
age_sex_dx.age = age_sex_dx.age.str.replace('(n\/a,)*(n\/a)?', '') # Removes the n/a unformatted
age_sex_dx = age_sex_dx[~age_sex_dx.age.isna() & (age_sex_dx.age.str.len()>0)] # Removes 4 participants
age_sex_dx.age = age_sex_dx.age.str.split(',').apply(lambda x: np.mean([int(e) for e in x]))
assert len(age_sex_dx) == 311 and age_sex_dx.participant_id.is_unique
assert np.all(~age_sex_dx.age.isna() & ~age_sex_dx.sex.isna())
age_sex_dx['site'] = 'NAR'
age_sex_dx['study'] = 'NAR'
age_sex_dx['diagnosis'] = 'control'
age_sex_dx.participant_id = age_sex_dx.participant_id.str.replace('sub-', '').astype(int).astype(str)
## Dataset-independent
tiv = pd.read_csv(os.path.join(NAR_path, 'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 4
age_sex_dx_site_study_tiv.to_csv(os.path.join(NAR_path, 'NAR_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make RBP phenotype
RBP_PATH = '/neurospin/psy_sbox/rbp'
## Dataset dependent
age_sex = pd.read_csv(os.path.join(RBP_PATH, 'participants.tsv'), sep='\t')
age_sex = age_sex.rename(columns = {'Age': 'age', 'Gender': 'sex', 'ID': 'participant_id'})
age_sex.sex = age_sex.sex.map({'M': 0, 'F': 1})
age_sex['study']='RBP'
age_sex['site']='RBP'
age_sex['diagnosis']='control'
age_sex.participant_id = age_sex.participant_id.astype(str)
assert age_sex.participant_id.is_unique
## Dataset-independent
tiv = pd.read_csv(os.path.join(RBP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str) # 9 bad segmentations with no TIV
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 2 # 2 participants not found in participants.tsv (18,40)
age_sex_dx_site_study_tiv.to_csv(os.path.join(RBP_PATH, 'RBP_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make OASIS 3 phenotype
OASIS3_PATH = '/neurospin/psy_sbox/oasis3'
## Dataset dependent
dx = pd.read_csv(os.path.join(OASIS3_PATH, 'adrc_clinical_data.csv'), sep=',')
sex = pd.read_csv(os.path.join(OASIS3_PATH, 'participants.tsv'), sep='\t')
age = pd.read_csv(os.path.join(OASIS3_PATH, 'MR_sessions.tsv'), sep='\t')
age = age.rename(columns={'Age': 'age', 'Subject': 'participant_id'})
age['session'] = age['MR ID'].str.extract('(OAS[0-9]+)\_MR\_(d[0-9]+)')[1]
sex = sex.rename(columns={'Subject': 'participant_id', 'M/F': 'sex'})
sex.sex = sex.sex.map({'M': 0, 'F': 1})
age_sex = pd.merge(age, sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex = age_sex[~age_sex.age.isna() & ~age_sex.sex.isna()]
dx = dx.rename(columns={'Subject': 'participant_id'})
dx.drop(columns=['Age', 'Date'], inplace=True)

# Selects only patients who kept their CDR constant (no CTL to AD or AD to CTL and AD is defined as CDR > 0)
# cf Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI Data in Young, Middle Aged, Nondemented,
# and Demented Older Adults, Daniel S. Marcus, Journal of Cognitive Neuroscience, 2007
cdr_min = dx.groupby('participant_id', as_index=False)['cdr'].agg(np.min)
cdr_max = dx.groupby('participant_id', as_index=False)['cdr'].agg(np.max)
constant_cdr_participants_id = cdr_min[cdr_min.cdr == cdr_max.cdr]
# In line with https://www.oasis-brains.org/files/OASIS-3_Imaging_Data_Dictionary_v1.8.pdf
assert (constant_cdr_participants_id.cdr==0).sum() == 605 and (constant_cdr_participants_id.cdr==0.5).sum() == 66
# Filters the participants
age_sex = age_sex[age_sex.participant_id.isin(constant_cdr_participants_id.participant_id)]
age_sex['diagnosis'] = 'control'
filter_ad = age_sex.participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr > 0].participant_id)
age_sex.loc[filter_ad, 'diagnosis'] = 'AD'
assert np.all(age_sex[age_sex.diagnosis.eq('control')].participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr==0].participant_id))
assert np.all(age_sex[age_sex.diagnosis.eq('AD')].participant_id.isin(constant_cdr_participants_id[constant_cdr_participants_id.cdr>0].participant_id))
age_sex['study'] = 'OASIS3'
age_sex['site'] = 'OASIS3'
## Dataset-independent
tiv = pd.read_csv(os.path.join(OASIS3_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on=['participant_id', 'session'], how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
age_sex_dx_site_study_tiv.to_csv(os.path.join(OASIS3_PATH, 'OASIS3_t1mri_mwp1_participants.csv'), sep='\t', index=False)


## Make GSP phenotype
GSP_PATH = '/neurospin/psy_sbox/GSP'
## Dataset dependent
age_sex = pd.read_csv(os.path.join(GSP_PATH, 'participants_ses-1.tsv'), sep='\t')
age_sex_rescanned = pd.read_csv(os.path.join(GSP_PATH, 'participants_ses-1_ses-2.tsv'), sep='\t')
age_sex['participant_id'] = age_sex.Subject_ID.str.extract('Sub([0-9]+)_*')
age_sex_rescanned['participant_id'] = age_sex_rescanned.Subject_ID.str.extract('Sub([0-9]+)_*')
# Ensures that we have all the data we need from age_sex
assert age_sex.participant_id.is_unique and set(age_sex.participant_id) >= set(age_sex_rescanned.participant_id)
assert np.all(~age_sex.Age_Bin.isna()) and np.all(~age_sex.Sex.isna())
age_sex = age_sex.rename(columns={'Age_Bin': 'age', 'Sex': 'sex'})
age_sex.sex = age_sex.sex.map({'M': 0, 'F': 1})
age_sex.participant_id = age_sex.participant_id.astype(int).astype(str)
age_sex['diagnosis'] = 'control'
age_sex['study'] = 'GSP'
age_sex['site'] = 'HUV'
## Dataset-independent
tiv = pd.read_csv(os.path.join(GSP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv)
age_sex_dx_site_study_tiv.to_csv(os.path.join(GSP_PATH, 'GSP_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make ICBM phenotype
ICBM_PATH = "/neurospin/psy_sbox/icbm"
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(ICBM_PATH, 'participants.tsv'), sep='\t')
age_sex_dx['participant_id'] = age_sex_dx['Subject'].str.replace('_','')
age_sex_dx = age_sex_dx.rename(columns={'Sex': 'sex', 'Age': 'age'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['diagnosis'] = 'control'
age_sex_dx.drop_duplicates(subset='participant_id', keep='first', inplace=True)
assert len(age_sex_dx.participant_id) == len(set(age_sex_dx.participant_id)) == 640
age_sex_dx['study'] = 'ICBM'
age_sex_dx['site'] = 'ICBM'
## Dataset independent
tiv = pd.read_csv(os.path.join(ICBM_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 982
age_sex_dx_site_study_tiv.to_csv(os.path.join(ICBM_PATH, 'ICBM_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make CNP phenotype
CNP_PATH = "/neurospin/psy_sbox/cnp"
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(CNP_PATH, 'participants.tsv'), sep='\t')
age_sex_dx.participant_id = age_sex_dx.participant_id.str.replace('sub-', '')
age_sex_dx.diagnosis = age_sex_dx.diagnosis.map({'CONTROL': 'control', 'SCHZ': 'schizophrenia', 'BIPOLAR': 'bipolar',
                                                 'ADHD': 'adhd'})
age_sex_dx = age_sex_dx.rename(columns={'gender': 'sex'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['site'] = 'CNP'
age_sex_dx['study'] = 'CNP'
assert age_sex_dx.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(CNP_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='1:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) == 265
age_sex_dx_site_study_tiv.to_csv(os.path.join(CNP_PATH, 'CNP_t1mri_mwp1_participants.csv'), sep='\t', index=False)


## Make LOCALIZER phenotype
LOCALIZER_PATH = "/neurospin/psy_sbox/localizer"
## Dataset dependent
age_sex_dx_site = pd.read_csv(os.path.join(LOCALIZER_PATH, 'participants.tsv'), sep='\t')
age_sex_dx_site.sex = age_sex_dx_site.sex.map({'M':0, 'F':1})
age_sex_dx_site['study'] = 'LOCALIZER'
age_sex_dx_site = age_sex_dx_site[~age_sex_dx_site.age.isna() & ~age_sex_dx_site.age.eq('None')] # 4 participants have 'None' age
age_sex_dx_site.age = age_sex_dx_site.age.astype(float)
age_sex_dx_site['diagnosis'] = 'control'
assert age_sex_dx_site.participant_id.is_unique
## Dataset independent
tiv = pd.read_csv(os.path.join(LOCALIZER_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx_site, on='participant_id', how='left', sort=False, validate='1:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 4
age_sex_dx_site_study_tiv.to_csv(os.path.join(LOCALIZER_PATH, 'LOCALIZER_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make MPI_LEIPZIG phenotype
MPI_PATH = '/neurospin/psy_sbox/mpi-leipzig'
## Dataset dependent
age_sex_dx = pd.read_csv(os.path.join(MPI_PATH, 'participants.tsv'), sep='\t')
age_sex_dx = age_sex_dx.rename(columns={'gender': 'sex', 'age (5-year bins)': 'age'})
age_sex_dx.sex = age_sex_dx.sex.map({'M': 0, 'F': 1})
age_sex_dx['participant_id'] = age_sex_dx['participant_id'].str.replace('sub-','').astype(int).astype(str)
assert len(age_sex_dx) == 318
age_sex_dx = age_sex_dx[~age_sex_dx.age.isna()]
assert len(age_sex_dx) == 316 and age_sex_dx.participant_id.is_unique
age_sex_dx.age = age_sex_dx.age.str.split('-').apply(lambda x: np.mean([int(e) for e in x]))
age_sex_dx['site'] = 'MPI-LEIPZIG'
age_sex_dx['study'] = 'MPI-LEIPZIG'
age_sex_dx['diagnosis'] = 'control'
## Dataset independent
tiv = pd.read_csv(os.path.join(MPI_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 2
age_sex_dx_site_study_tiv.to_csv(os.path.join(MPI_PATH, 'MPI-LEIPZIG_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make CANDI phenotype
CANDI_PATH = '/neurospin/psy_sbox/candi'
age_sex_dx = pd.read_csv(os.path.join(CANDI_PATH, 'SchizBull_2008_Demographics_V1.1.csv'), sep=',')
age_sex_dx = age_sex_dx.rename(columns={'Gender': 'sex', 'Age': 'age'})
age_sex_dx['participant_id'] = age_sex_dx.Subject.str.replace('_', '')
age_sex_dx['diagnosis'] = age_sex_dx.Subject.str.extract('(\w+)\_[0-9]+')[0]
age_sex_dx.diagnosis = age_sex_dx.diagnosis.map({'HC': 'control', 'BPDwoPsy': 'bipolar disorder without psychosis',
                                                 'BPDwPsy': 'bipolar disorder with psychosis', 'SS': 'schizophrenia'})
age_sex_dx.sex = age_sex_dx.sex.map({'male': 0, 'female': 1})
age_sex_dx['site'] = 'CANDI'
age_sex_dx['study'] = 'CANDI'
## Dataset independent
tiv = pd.read_csv(os.path.join(CANDI_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex_dx, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna() &
                                                      ~age_sex_dx_site_study_tiv.diagnosis.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv)
age_sex_dx_site_study_tiv.to_csv(os.path.join(CANDI_PATH, 'CANDI_t1mri_mwp1_participants.csv'), sep='\t', index=False)

## Make NPC phenotype
NPC_PATH = '/neurospin/psy_sbox/npc'
age_sex = pd.read_csv(os.path.join(NPC_PATH, 'participants.tsv'), sep='\t')
age_sex.sex = age_sex.sex.map({'male': 0, 'female': 1})
age_sex = age_sex[~age_sex.sex.isna()] # 1 NaN
age_sex.participant_id = age_sex.participant_id.str.replace('sub-', '').astype(int).astype(str)
age_sex['diagnosis'] = 'control'
age_sex['site'] = 'NPC'
age_sex['study'] = 'NPC'
## Dataset independent
tiv = pd.read_csv(os.path.join(NPC_PATH,'derivatives/cat12-12.6_vbm_roi/cat12-12.6_vbm_roi.tsv'), sep='\t')
tiv.participant_id = tiv.participant_id.astype(str)
## Merge all
age_sex_dx_site_study_tiv = pd.merge(tiv, age_sex, on='participant_id', how='left', sort=False, validate='m:1')
age_sex_dx_site_study_tiv = age_sex_dx_site_study_tiv[~age_sex_dx_site_study_tiv.age.isna() &
                                                      ~age_sex_dx_site_study_tiv.sex.isna() &
                                                      ~age_sex_dx_site_study_tiv.TIV.isna()]
assert len(age_sex_dx_site_study_tiv) == len(tiv) - 1 # 1 participant sex missing
age_sex_dx_site_study_tiv.to_csv(os.path.join(NPC_PATH, 'NPC_t1mri_mwp1_participants.csv'), sep='\t', index=False)
