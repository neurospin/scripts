# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 12:13:38 2014

@author: hl237680

Merging of two dataframes in order to get a new .csv file containing clinical
data (initial .xls file from the SHFJ) but also additional parameters such as
gestational duration and socio-economic factor for the IMAGEN cohort.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"
    clinical data on IMAGEN population (.xls initial data file)
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/
    socio_eco_factor_and_gestational_time.xls":
    additional factors for the IMAGEN cohort

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/more_clinics.csv:"
    .csv file containing both traditional clinical data but also additional
    parameters such as gestational duration and socio-economic factor

"""


import os
import xlrd
import numpy as np
import pandas as pd


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SHFJ_DATA_PATH = os.path.join(CLINIC_DATA_PATH, 'source_SHFJ')


# Excel file containing clinical data of the 1.265 subjects for whom we
# have both neuroimaging and genetic data
workbook1 = xlrd.open_workbook(os.path.join(SHFJ_DATA_PATH,
                                           '1534bmi-vincent2.xls'))
# Dataframe with right indexes
demographics_df = pd.io.excel.read_excel(workbook1,
                                         sheetname='1534bmi-gaser.xls',
                                         engine='xlrd',
                                         header=0,
                                         index_col=0)

# Excel file containing additional data such as socio-economic factor and
# time of pregnancy
workbook2 = xlrd.open_workbook(os.path.join(CLINIC_DATA_PATH,
                                 'socio_eco_factor_and_gestational_time.xls'))
# Dataframe with right indexes
add_factors_df = pd.io.excel.read_excel(workbook2,
                                        sheetname='additional_factors',
                                        engine='xlrd',
                                        header=0,
                                        index_col=0)

# Intercept of the subjects whose data are stored into both dataframes
subjects_id = np.intersect1d(demographics_df.index.values,
                             add_factors_df.index.values).tolist()

demographics_df = demographics_df.loc[subjects_id]
add_factors_df = add_factors_df.loc[subjects_id]

## Cofounds of non interest (dummy coding for categorical variables)
#cofounds1 = ['Gender',
#             'tiv_gaser',
#             'mean_pds']

sup_df = pd.merge(demographics_df, add_factors_df,
                  left_index=True, right_index=True)

# Write more complete dataframe into a .csv file
sup_df.to_csv(os.path.join(CLINIC_DATA_PATH,
                           'more_clinics.csv'))