# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 17:34:58 2014

@author: hl237680

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects with normal status, for further use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI_norm_group.phe"

Only normal teenagers (i.e. 910 subjects) among the 1.265 subjects for whom
we have both neuroimaging and genetic data have been selected.

"""


import os
import csv
import numpy as np
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SHFJ_DATA_PATH = os.path.join(CLINIC_DATA_PATH, 'source_SHFJ')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')
PLINK_GENETICS_PATH = os.path.join(GENETICS_PATH, 'Plink')
BMI_PHENOTYPE = os.path.join(PLINK_GENETICS_PATH, 'BMI_norm_group.phe')

# Generation of a dataframe containing clinical data to establish a .phe file
print '###################################################'
print '# Generation of a .phe file for the BMI phenotype #'
print '###################################################'

# Dataframe with right clinical data
clinical_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                            'clinical_data_norm_group.csv'),
                                      sep=',',
                                      index_col=0)

cofounds = ['BMI']

# Keep only overweight and obese people among subjects for whom we have all
# data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                            dtype=None, delimiter=',', skip_header=1)
subjects_index = np.intersect1d(subjects_id.tolist(), clinical_df.index.values)

# Dataframe containing only non interest covariates for selected subjects
dataframe = pd.DataFrame(clinical_df, index=subjects_index, columns=cofounds)

# Write .phe file
fp = open(BMI_PHENOTYPE, 'wb')
cw = csv.writer(fp, delimiter=' ')
cw.writerow(['FID', 'IID', 'BMI'])
for i, s in enumerate(subjects_index):
    tmp = []
    # Family ID (FID)
    tmp.append('%012d' % (int(s)))
    # Individual ID (IID)
    tmp.append('%012d' % (int(s)))
    # BMI
    tmp.append('%.2f' % (dataframe.loc[s]))
    cw.writerow(tmp)
fp.close()