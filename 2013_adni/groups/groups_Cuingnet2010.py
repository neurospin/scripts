# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:12:27 2013

@author: md238665

This script reproduces the groups of Cuingnet 2010
and the groups in BV database
from data in ADNImerge.

"""

import os

import numpy
import pandas

DOC_PATH = "/neurospin/cati/ADNI/ADNI_510/documents"
INPUT_REF_FILE = os.path.join(DOC_PATH, "subjects_diagnostics_list.txt")
CLINIC_PATH = "/neurospin/brainomics/2013_adni_preprocessing/clinic"
INPUT_ADNIMERGE = os.path.join(CLINIC_PATH, "adnimerge.csv")
OUTPUT_GROUPS = 'groups.csv'

# Read ADNI file
adni = pandas.read_csv(INPUT_ADNIMERGE)

# Read subjects used by Rémy
ref_file = pandas.read_table(INPUT_REF_FILE, sep=" ",
                                  header=None, names=['Ref', 'Group'])
ref_file.index = ref_file['Ref'].map(lambda x: x[0:10])

# Merge them
adni_ref = pandas.merge(adni, ref_file, left_on='PTID', right_index=True)

# Some visit don't have a diagnosis so remove them
# This doesn't affect baseline
null_indexes = adni_ref['DX'].isnull()
adni_ref = adni_ref[~null_indexes]

bl_indexes = (adni_ref['EXAMDATE'] == adni_ref['EXAMDATE.bl'])

# Recode DX.bl: this should give the group
adni_ref['DX.bl'] = adni_ref['DX.bl'].map(
  {'AD': 'AD',
   'LMCI': 'MCI',
   'CN': 'control'})
# PTID is used for index
bv_group = adni_ref[['PTID', 'DX.bl']][bl_indexes].copy()
bv_group.index = bv_group['PTID']

# Compare them
group_cmp = (bv_group['DX.bl'] == adni_ref['Group'][bl_indexes])
if group_cmp.all():
    print "Youhou"
else:
    i = numpy.where(~group_cmp)[0]
    print "Oh non: subject(s)", group_cmp.index[i], "differ"

# Column DX is differentially coded
# Therefore I recode it
adni_ref['DX'] = adni_ref['DX'].map(
  {'Dementia': 'AD',
   'MCI': 'MCI',
   'NL':   'control',
   'NL to MCI': 'MCI',
   'NL to Dementia': 'AD',
   'MCI to Dementia': 'AD',
   'Dementia to MCI': 'MCI',
   'MCI to NL': 'control'},
   na_action='ignore')
adni_ref['DX'][bl_indexes] = adni_ref['DX.bl'][bl_indexes]

## Find 18-month converters with a loop
a = adni_ref['PTID'].unique()
converters_PTID = []
for ptid in a:
    subject_lines = adni_ref[adni_ref['PTID'] == ptid]
    subject_lines_before18m = subject_lines[subject_lines['M'] <= 18]
    initial_dx = subject_lines_before18m['DX'].iloc[0]
    last_dx = subject_lines_before18m['DX'].iloc[-1]
    if (initial_dx == 'MCI') and (last_dx == 'AD'):
        #print "Adding", ptid
        #print subject_lines_before18m[['M', 'DX']]
        converters_PTID.append(ptid)
#
Cuignet2010_group = adni_ref[['PTID', 'DX']][bl_indexes].copy()
Cuignet2010_group['DX'] = Cuignet2010_group['DX'].map(
  {'AD': 'AD',
   'MCI': 'MCInc',
   'control': 'control'})
Cuignet2010_group['DX'][Cuignet2010_group['PTID'].isin(converters_PTID)] = 'MCIc'
Cuignet2010_group.index = Cuignet2010_group['PTID']
#print Cuignet2010_group['DX'].value_counts()

group_cmp = pandas.DataFrame.from_items(
  [('brainvisa_group', bv_group['DX.bl']),
   ('Cuignet2010_group', Cuignet2010_group['DX'])])
group_cmp.sort().to_csv(OUTPUT_GROUPS)
