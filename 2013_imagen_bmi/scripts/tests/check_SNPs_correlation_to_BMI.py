# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 17:47:30 2014

@author: hl237680

This script aims at comparing SNPs referenced in the literature as
associated to BMI and all SNPs extracted from the genes known to be
associated to BMI.

INPUT:
- SNPs referenced in the literature as associated to BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    SNPs_BMI_ob_literature.xls"
- all SNPs extracted from the genes known to be associated to BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/BMI_SNPs_names_list.csv"

OUTPUT: intercept of both lists of SNPs
    SNPs_of_interest

"""

import os
import xlrd
import numpy as np


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')

# Data on SNPs extracted from the literature on BMI and genetics
workbook = xlrd.open_workbook(os.path.join(GENETICS_PATH,
                                           'SNPs_BMI_ob_literature.xls'))
sheet = workbook.sheet_by_name('SNPs_from_literature')

SNPs_from_literature = []
n_rows = 66
for row in range(n_rows):
    SNPs_from_literature.append(sheet.cell(row, 0).value)
    SNPs_from_literature[row].replace("text:u", "").replace("'", "")

SNPs_from_literature = [str(snp) for snp in SNPs_from_literature]

# All SNPs extracted from genes associated with BMI (UCSC)
SNPs_from_BMI_genes = np.genfromtxt(os.path.join(DATA_PATH,
                                                 'BMI_SNPs_names_list.csv'),
                                    dtype=None).tolist()

SNPs_of_interest = np.intersect1d(SNPs_from_literature,
                                  SNPs_from_BMI_genes)