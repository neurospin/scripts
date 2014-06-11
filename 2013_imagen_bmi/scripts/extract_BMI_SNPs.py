# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:43:10 2014

@author: hl237680
"""

import os
import numpy as np
import pandas as pd
import csv


# Input data
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GENETICS_DATA_PATH = os.path.join(DATA_PATH, 'genetics')


##########################################################
# Read subject list, BMI and covariates for all subjects #
##########################################################

CLINIC_FILE = pd.io.parsers.read_csv(os.path.join(CLINIC_DATA_PATH, "1534bmi-vincent2.csv")).as_matrix()
CLINIC_COV_DIR = os.path.join(CLINIC_DATA_PATH, 'clinical_cov.csv')
#clinic_subjects_id = CLINIC_FILE[:,0]
#gender = CLINIC_FILE[:,1]
#age_in_year = CLINIC_FILE[:,2]
#pds_status = CLINIC_FILE[:,30]
#BMI = CLINIC_FILE[:,8]


#clinical_cov = np.zeros((CLINIC_FILE.shape[0], 5))
#clinical_cov[:,0] = CLINIC_FILE[:,0]
#clinical_cov[:,1] = CLINIC_FILE[:,1]
#clinical_cov[:,2] = CLINIC_FILE[:,2]
#clinical_cov[:,3] = CLINIC_FILE[:,30]
#clinical_cov[:,4] = CLINIC_FILE[:,8]

c = csv.writer(open("(os.path.join(CLINIC_DATA_PATH, 'clinical_cov.csv')", "wb"))
c.writerow(["ID", "Gender", "Age in year", "PDS", "BMI"])
for i in np.arange(0, CLINIC_FILE.shape[0]-1, 1):
    c.writerow([CLINIC_FILE[i,0], CLINIC_FILE[i,1], CLINIC_FILE[i,2], CLINIC_FILE[i,30], CLINIC_FILE[i,8]])


    

#GENETICS_FILE = pd.io.parsers.read_csv(os.path.join(GENETICS_DATA_PATH, "bmi_snp.csv")).as_matrix()
#SNPs_subjects_id = ???
#
## We select subjects for which we have:
##  - genetics data
##  - an image quality estimated as A or B after quality control
#subjects_id_gen_image = set(SNPs_subjects_id).intersection(set(clinic_subjects_id))
##subjects_id = subset(tab, Subjects%in%subjects_id_gen_image & quality_control != 'C')$Subjects
##subjects_id_char = as.character(subjects_id)
#
#
