# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 16:57:06 2014

@author: hl237680

This script aims at generating a csv file from various xls files in
order to gather all useful clinical data.

INPUT: xls files
CLINIC_DATA_PATH: /neurospin/brainomics/2013_imagen_bmi/data/clinic/
SHFJ_DATA_PATH: /neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    - SHFJ_DATA_PATH: 1534bmi-vincent2.xls: initial data file.
    - CLINIC_DATA_PATH: BMI_status.xls: file containing reference values to
    attribute a weight status (i.e. Insuff, Normal, Overweight, Obese) to
    subjects according to gender, age and their BMI.

OUTPUT: csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"

Only the 1265 subjects for which we have neuroimaging data (masked_images)
are selected among the total number of subjects.

(Possibility to select subjects according to their status.)
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
IMAGES_FILE = os.path.join(DATA_PATH, 'smoothed_images.hdf5')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')


############################################################################
# Attribute a status to adolescents according to their gender, age and BMI #
############################################################################

def status_col(subject, ref_values):
    # For each subject, read age:
    age = subject['ageannees']
    # gender:
    gender = subject['Gender de Feuil2']
    # BMI:
    bmi = subject['BMI']
    # Knowing  these parameters -to be compared to reference values (OMS)-
    # determine weight status:
    index = ((age >= ref_values['Min Age']) & (age < ref_values['Max Age'])
                & (ref_values['Gender'] == gender))
    LB = ref_values[['Normal', 'Overweight', 'Obese']][index].as_matrix()
    #print index
    lb_normal = LB[0, 0]
    lb_overweight = LB[0, 1]
    lb_obese = LB[0, 2]
    if (bmi < lb_normal):
        return 'Insuff'
    if (bmi >= lb_normal) & (bmi < lb_overweight):
        return 'Normal'
    if (bmi >= lb_overweight) & (bmi < lb_obese):
        return 'Overweight'
    if (bmi >= lb_obese):
        return 'Obese'


if __name__ == "__main__":
    # Generation of a csv file containing all clinical data available
    print '###################################################'
    print '# Generation of a csv file from various xls files #'
    print '###################################################'

    # Excel files containing clinical data to be read
    workbook = xlrd.open_workbook(os.path.join(SHFJ_DATA_PATH,
                                               '1534bmi-vincent2.xls'))
    #workbook2 = xlrd.open_workbook(os.path.join(SHFJ_DATA_PATH,
    #                                           "bmi-tri.xls"))

    # Dataframes with right indexes
    df = pd.io.excel.read_excel(workbook, sheetname='1534bmi-gaser.xls',
                                engine='xlrd', header=0, index_col=0)
    #df2 = pd.io.excel.read_excel(workbook2, sheetname='bmi-tri.xls',
    #                             engine='xlrd', header=0, index_col=0)

    # Cofounds: non interest covariates for each file
    cofounds = ['Gender de Feuil2',  # Gender: Male or Female (instead of -1/1)
                'Age...sstartdate',
                'ageannees',
                'ImagingCentreID',
                'ImagingCentreCity',
                'NI_MASS',
                'NI_HEIGHT',
                'BMI',
                'GE',
                'PH',
                'SIE',
                'quality_control',
                'bad_cov_gaser',
                'qc_gaser',
                'Age.for.timestamp',
                'total_f',
                'mean_f',
                'total_m',
                'mean_m',
                'mean_pds',
                'PDS-cg',
                'vol_gm_gaser',
                'vol_wm_gaser',
                'vol_csf_gaser',
                'tiv_gaser'
                ]

    # Keep only subjects for which we have all data
    subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                                dtype=None, delimiter=',', skip_header=1)

    # Dataframe containing only non interest covariates for selected subjects
    dataframe = pd.DataFrame(df, index=subjects_id, columns=cofounds)

    # Excel file with reference BMI status according to age and gender
    workbook = xlrd.open_workbook(os.path.join(CLINIC_DATA_PATH,
                                               'BMI_status.xls'))

    # Create a dataframe assigning status to subjects
    ref_values = pd.io.excel.read_excel(workbook,
                                        sheetname='ref_values_OMS_2007',
                                        engine='xlrd',
                                        header=0)
    # Apply function returns a panda.core.series
    status_series = dataframe.apply(status_col, args=(ref_values,), axis=1)
    # Conversion to a dataframe
    status_df = pd.DataFrame(status_series, index=subjects_id)
    # Rename column giving the weight status
    status_df = status_df.rename(columns={0: 'Status'})

    # Merge both dataframes
    clinical_data = pd.merge(dataframe, status_df,
                             left_index=True,
                             right_index=True)

    clinical_data_all = pd.DataFrame.to_csv(clinical_data,
                            os.path.join(CLINIC_DATA_PATH, 'population.csv'))

    print "CSV file containing all clinical data has been saved."


#print '###################'
#print '# Group selection #'
#print '###################'
#
#overweight_group = clinical_data[clinical_data['Status'] == 'Overweight']
#obese_group = clinical_data[clinical_data['Status'] == 'Obese']
#o_o_groups = overweight_group.append(obese_group, ignore_index=False)
#overweight_obese_group_file = pd.DataFrame.to_csv(o_o_groups,
#                                os.path.join(CLINIC_DATA_PATH,
#                                'clinical_data_overweight_obese_group.csv'))
#
#print "CSV file containing clinical data from overweight and obese status subjects has been saved."