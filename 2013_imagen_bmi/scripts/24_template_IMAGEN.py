# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:40:45 2014

@author: hl237680

Work on the template provided by the SHFJ and used to process our data.
Take a look at the demographics of the subjects whose images were chosen to
draw this template.

INPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
      1534bmi-vincent2.xls"
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/BMI_status.xls"
    "/neurospin/brainomics/2013_imagen_bmi/data/template/
      adresses_images_240_template.txt"

OUTPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/template/
      template_IMAGEN_clinics.csv"
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
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')
TEMPLATE_PATH = os.path.join(DATA_PATH, 'template')


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
    print '########################################################'
    print '# Generation of a csv file containing clinical data and weight status for the 1534 subjects involved in the IMAGEN study #'
    print '########################################################'

    # Excel files containing clinical data to be read
    workbook = xlrd.open_workbook(os.path.join(SHFJ_DATA_PATH,
                                               '1534bmi-vincent2.xls'))

    # Dataframes with right indexes
    df = pd.io.excel.read_excel(workbook, sheetname='1534bmi-gaser.xls',
                                engine='xlrd', header=0, index_col=0)

    # Parameters of interest
    cofounds = ['Gender de Feuil2',
                'ageannees',
                'ImagingCentreCity',
                'NI_MASS',
                'NI_HEIGHT',
                'BMI',
                'mean_pds',
                ]

    # Dataframe containing only non interest covariates for selected subjects
    dataframe = pd.DataFrame(df, columns=cofounds)

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
    status_df = pd.DataFrame(status_series)

    # Rename column giving the weight status
    status_df = status_df.rename(columns={0: 'Status'})

    # Merge both dataframes
    clinical_data = pd.merge(dataframe,
                             status_df,
                             left_index=True,
                             right_index=True)

    IMAGEN_1534subj = pd.DataFrame.to_csv(clinical_data,
                         os.path.join(CLINIC_DATA_PATH, 'IMAGEN_1534subj.csv'))

    print "CSV file containing clinical data and weight status for"
    print "the 1534 subjects involved in the IMAGEN study has been saved."

    # Consider images used to build the IMAGEN template
    images_for_template = np.genfromtxt(os.path.join(TEMPLATE_PATH,
                                          'adresses_images_240_template.txt'),
                                        dtype=None,
                                        delimiter=' ')

    # Consider subjects whose images were used to build the IMAGEN template
    template_subjects_id = []
    for i, image_id in enumerate(images_for_template):
        subject_id = images_for_template[i][len('/home/Pgipsy/IMAGENERIE/'
                                                'Morpho/VBM/images/'
                                                'original/'):
                                            -len('s005a1001.nii')]
        if len(subject_id) == 12:
            subject_id = int(subject_id)
        else:
            subject_id = int(subject_id[:-1])

        template_subjects_id.append(subject_id)

    # Load clinical data from the IMAGEN subjects as a dataframe
    # 1265 subjects for whom we have both neuroimaging and genetic data
    population_df = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH,
                                                        'IMAGEN_1534subj.csv'),
                                           index_col=0)

    # Parameters of interest
    colnames = ['Gender de Feuil2', 'ageannees', 'BMI', 'Status']

    # Dataframe containing data of interest for the selected subjects whose
    # images were used to build the IMAGEN template
    template_subjects_clinics_df = pd.DataFrame(population_df,
                                                index=template_subjects_id,
                                                columns=colnames)

    template_IMAGEN_subjects_clinics = pd.DataFrame.to_csv(
                                            template_subjects_clinics_df,
                                            os.path.join(TEMPLATE_PATH,
                                                'template_IMAGEN_clinics.csv'))
    #
    #
    ## Demographics
    #insuff_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Insuff']
    #normal_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Normal']
    #overweight_group = full_sulci_clinic_df[full_sulci_clinic_df
    #                                            ['Status'] == 'Overweight']
    #obese_group = full_sulci_clinic_df[full_sulci_clinic_df['Status'] == 'Obese']
    #
    #male_group = full_sulci_clinic_df[full_sulci_clinic_df
    #                                            ['Gender de Feuil2'] == 'Male']
    #female_group = full_sulci_clinic_df[full_sulci_clinic_df
    #                                            ['Gender de Feuil2'] == 'Female']
    #
    #print "There are ", insuff_group.shape[0], "thick people."
    #print "There are ", normal_group.shape[0], "normal people."
    #print "There are ", overweight_group.shape[0], "overweight people."
    #print "There are ", obese_group.shape[0], "obese people."
    #
    #print "There are ", male_group.shape[0], "male people."
    #print "There are ", female_group.shape[0], "female people."