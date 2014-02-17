# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:39:00 2013

@author: md238665
"""

import os

import pandas

DOC_PATH="/neurospin/cati/ADNI/ADNI_510/documents"
INPUT_REF_FILE=os.path.join(DOC_PATH, "subjects_diagnostics_list.txt")

# Read ADNI file
CLINIC_PATH="/neurospin/brainomics/2013_adni_preprocessing/clinic"
INPUT_ADNIMERGE=os.path.join(CLINIC_PATH, "adnimerge.csv")
adni = pandas.read_csv(INPUT_ADNIMERGE)

# Index it
adni['EXAMDATE'] = pandas.to_datetime(adni['EXAMDATE'])
adni['EXAMDATE.bl'] = pandas.to_datetime(adni['EXAMDATE.bl'])

subject_data_index = pandas.MultiIndex.from_arrays([adni['PTID'], adni['EXAMDATE']])
adni.index = subject_data_index

# Only baseline examination
adni_bl = adni.loc[adni['EXAMDATE'] == adni['EXAMDATE.bl']]
adni_bl.index = adni_bl['PTID']

# Read subjects used by Rémy
remy_subjects = pandas.read_table(INPUT_REF_FILE, sep=" ",
                                  header=None, names=['Ref', 'Group'])
remy_subjects.index = remy_subjects['Ref'].map(lambda x: x[0:10])

# Merge them
adni_bl_remy = pandas.merge(adni_bl, remy_subjects, left_index=True, right_index=True)

# Check if baseline diagnosis is consistent with diagnostic
DX_bl = adni_bl_remy['DX.bl'].map({'AD':   'AD',
                              'LMCI': 'MCI',
                              'EMCI': 'MCI',
                              'SMC':  'control',
                              'CN':   'control'})

DX = adni_bl_remy['DX'].map({'Dementia': 'AD',
                        'MCI': 'MCI',
                        'NL':   'control',
                        'MCI to Dementia': 'AD',
                        'Dementia to MCI': 'MCI',
                        'MCI to NL': 'control'},
                        na_action='ignore')

comp = (DX_bl == DX)
if comp.all():
    print "OK"
