# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:57:06 2014

@author: hl237680

1 - This script aims at generating a csv file from various xls files in
order to gather all useful clinical data.
2 - Possibility to select subjects according to their status
(INSUF, Normal, ob1, ob2)
"""


import os
import xlrd
import numpy as np
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')



print '###################################################'
print '# Generation of a csv file from various xls files #'
print '###################################################'

# Excel files containing clinical data to be read
workbook1 = xlrd.open_workbook(os.path.join(CLINIC_DATA_PATH,
                                           "1534bmi-vincent2.xls"))
workbook2 = xlrd.open_workbook(os.path.join(CLINIC_DATA_PATH,
                                           "bmi-tri.xls"))

# Dataframes with right indexes
df1 = pd.io.excel.read_excel(workbook1, sheetname='1534bmi-gaser.xls',
                             engine='xlrd', header=0, index_col=0)
df2 = pd.io.excel.read_excel(workbook2, sheetname='bmi-tri.xls',
                             engine='xlrd', header=0, index_col=0)

# Cofounds: non interest covariates for each file
cofound1 = ["Gender de Feuil2", "ImagingCentreCity", "tiv_gaser", "mean_pds"]
cofound2 = ["STATUS"]

# Keep only subjects for which we have all data
subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"),
                            dtype=None, delimiter=',', skip_header=1)

# Dataframes containing only non interest covariates for selected subjects                            
dataframe1 = pd.DataFrame(df1, index=subjects_id, columns=cofound1)
dataframe2 = pd.DataFrame(df2, index=subjects_id, columns=cofound2)


all_data = pd.merge(dataframe1, dataframe2, right_index=True, left_index=True)

clinical_data_all = pd.DataFrame.to_csv(all_data,
                    os.path.join(CLINIC_DATA_PATH, "clinical_data_all.csv"))

print "CSV file containing all clinical data we are interested in has been saved."



print '###################'
print '# Group selection #'
print '###################'

normal_group = all_data[all_data['STATUS'] == 'Normal']
normal_group_file = pd.DataFrame.to_csv(normal_group,
                    os.path.join(CLINIC_DATA_PATH, "normal_group.csv"))

print "CSV file containing clinical data from normal status subject has been saved."