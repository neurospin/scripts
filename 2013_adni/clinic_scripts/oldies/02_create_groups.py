# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:12:27 2013

@author: md238665

This script reproduces the groups of Cuingnet 2010
and the groups in BV database
from data in ADNImerge.

"""

import os

import numpy as np
import pandas

DOC_PATH = "/neurospin/cati/ADNI/ADNI_510/documents"
INPUT_BV_FILE = os.path.join(DOC_PATH, "subjects_diagnostics_list.txt")
INPUT_REF_FILE = os.path.join(DOC_PATH,
                              "Subjects_Paper_Cuingnet2010",
                              "Groups_Cuingnet2010.csv")

CLINIC_PATH = "/neurospin/brainomics/2013_adni/clinic"
INPUT_ADNIMERGE = os.path.join(CLINIC_PATH, "adnimerge.csv")

OUTPUT_PATH = "/neurospin/brainomics/2013_adni/clinic"
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "adni510_groups.csv")

# Read ADNI file
adni = pandas.read_csv(INPUT_ADNIMERGE)

# Read subjects used by Cuingnet et al. 2010 (only 509 subjects)
ref_file = pandas.read_csv(INPUT_REF_FILE, index_col=0)
REF_FILE_MAP = {'CN': 'control',
                'MCIc': 'MCIc',
                'MCInc': 'MCInc',
                'AD': 'AD'}
ref_file['Group.article'] = ref_file['Group.article'].map(REF_FILE_MAP)

# Read subjects used by BV (same than Cuingnet plus the missing subject)
bv_file = pandas.read_table(INPUT_BV_FILE, sep=" ",
                            header=None, names=['PTID', 'Group.BV'])
bv_file.index = bv_file['PTID'].map(lambda x: x[0:10])

# Extract subjects in ADNI 510
adni_510_subjects = bv_file.index

# Subjects in ADNI 509
adni_509_subjects = ref_file.index

# Missing subject
missing_subject = list(set(adni_510_subjects) - set(adni_509_subjects))[0]
print "Subject", missing_subject, "is not in Cuingnet et al. 2010"

# Add the missing subject in ref_file
tmp = pandas.DataFrame([None], columns=['Group.article'])
tmp.index = pandas.Series([missing_subject], name='PTID')
ref_file = ref_file.append(tmp)

# Subsample ADNI (this include several examination)
adni_510 = adni[adni['PTID'].isin(adni_510_subjects)]

# Extract baseline examinations indexes
bl_indexes = (adni_510['EXAMDATE'] == adni_510['EXAMDATE.bl'])

# Recode DX.bl: this should give the group
adni_510['DX.bl'] = adni_510['DX.bl'].map(
  {'AD': 'AD',
   'LMCI': 'MCI',
   'CN': 'control'})

# Column DX is differentially coded
# Therefore I recode it in the whole dataset
adni_510['DX'] = adni_510['DX'].map(
  {'Dementia': 'AD',
   'MCI': 'MCI',
   'NL':   'control',
   'NL to MCI': 'MCI',
   'NL to Dementia': 'AD',
   'MCI to Dementia': 'AD',
   'Dementia to MCI': 'MCI',
   'MCI to NL': 'control'},
   na_action='ignore')
# Copy baseline values
adni_510['DX'][bl_indexes] = adni_510['DX.bl'][bl_indexes]

# Subsample adni510 (this dataframe is alignable with bv_file)
adni_510_bl = adni_510[bl_indexes].copy()
adni_510_bl.index = adni_510_bl['PTID']

# Compare adni_510_bl and bv_file
# TODO: is it better to merge them (on PTID) for comparison?
for ID in adni_510_subjects:
    if adni_510_bl['DX.bl'].loc[ID] != bv_file['Group.BV'].loc[ID]:
        print "Subject", ID, "differ in ADNI and brainvisa"

# Find 18-month converters with a loop
converters_PTID = []
for ptid in adni_510_subjects:
    subject_lines = adni_510[adni_510['PTID'] == ptid]
    subject_lines_before18m = subject_lines[subject_lines['M'] <= 18]
    initial_dx = subject_lines_before18m['DX'].iloc[0]
    last_dx = subject_lines_before18m['DX'].iloc[-1]
    if (initial_dx == 'MCI') and (last_dx == 'AD'):
        #print "Adding", ptid
        #print subject_lines_before18m[['M', 'DX']]
        converters_PTID.append(ptid)
adni_510_bl['DX'] = adni_510_bl['DX'].map(
  {'AD': 'AD',
   'MCI': 'MCInc',
   'control': 'control'})
adni_510_bl['DX'][adni_510_bl['PTID'].isin(converters_PTID)] = 'MCIc'

# Create a dataframe with all the groups
COLS = ['Group.article', 'Group.BV', 'Group.ADNI', 'Sample']
group_cmp = pandas.DataFrame(np.empty((510, len(COLS)), dtype='object'),
                                      index = adni_510_subjects)
group_cmp.columns = COLS
for ID in adni_510_subjects:
    group_cmp['Group.article'][ID] = ref_file['Group.article'].loc[ID]
    group_cmp['Group.BV'][ID] = bv_file['Group.BV'].loc[ID]
    group_cmp['Group.ADNI'][ID] = adni_510_bl['DX'].loc[ID]
    group_cmp['Sample'][ID] = ref_file['Sample'].loc[ID]

# Set the missing subject to 'testing'
group_cmp['Sample'].loc[missing_subject] = 'testing'

group_cmp.sort().to_csv(OUTPUT_FILE)
