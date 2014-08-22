# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:14:43 2014

@author: hl237680

This script aims at generating a .phe file, that is a dataframe with FID,
IID and BMI of IMAGEN subjects of interest, for further use with Plink.

INPUT: .xls initial data file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"

OUTPUT: .phe file
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/BMI.phe"

Only the 1265 subjects for which we have neuroimaging data (masked_images)
are selected among the total number of subjects.
"""


import os
import xlrd
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
BMI_PHENOTYPE = os.path.join(PLINK_GENETICS_PATH, 'BMI.phe')

# Generation of a dataframe containing clinical data to establish a .phe file
print '###################################################'
print '# Generation of a .phe file for the BMI phenotype #'
print '###################################################'

# Excel files containing clinical data to be read
workbook = xlrd.open_workbook(os.path.join(SHFJ_DATA_PATH,
                                           '1534bmi-vincent2.xls'))
# Dataframe with right indexes
df = pd.io.excel.read_excel(workbook, sheetname='1534bmi-gaser.xls',
                            engine='xlrd', header=0, index_col=0)

cofounds = ['BMI']

# Keep only subjects for which we have all data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                            dtype=None, delimiter=',', skip_header=1)
subjects_id_list = subjects_id.tolist()

# Dataframe containing only non interest covariates for selected subjects
dataframe = pd.DataFrame(df, index=subjects_id, columns=cofounds)

# Write .phe file
fp = open(BMI_PHENOTYPE, 'wb')
cw = csv.writer(fp, delimiter=' ')
cw.writerow(['FID', 'IID', 'BMI'])
for i, s in enumerate(subjects_id_list):
    tmp = []
    # Family ID (FID)
    tmp.append('%012d' % (int(s)))
    # Individual ID (IID)
    tmp.append('%012d' % (int(s)))
    # BMI
    tmp.append('%.2f' % (dataframe.loc[s]))
    cw.writerow(tmp)
fp.close()