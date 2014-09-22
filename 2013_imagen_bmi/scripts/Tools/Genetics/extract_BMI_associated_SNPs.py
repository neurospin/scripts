# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 17:47:30 2014

@author: hl237680

This script aims at comparing SNPs referenced in the literature as
associated to BMI and all SNPs extracted from the genes known to be
associated to BMI.

INPUT:
- SNPs referenced in the literature as associated to BMI (included in or near
to BMI-associated genes):
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    genes_BMI_ob.xls"
- all SNPs extracted from the genes known to be associated to BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/BMI_SNPs_names_list.csv"

OUTPUT:
- SNPs_of_interest: intercept of both lists of SNPs
    "neurospin/brainomics/2013_imagen_bmi/data/genetics/BMI_SNPs_intercept.csv"
- SNPs distribution among IMAGEN subjects:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/SNPs_distribution.txt"

Results: only 10 SNPs are both included in BMI genes and explicitely
         candidate SNPs (see the literature) included in or near to
         BMI-associated genes.

"""

import os
import xlrd
import numpy as np
import pandas as pd
import csv


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')

# Output results
OUTPUT_DIR = os.path.join(GENETICS_PATH, 'Results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Data on SNPs extracted from the literature on BMI and genetics
workbook = xlrd.open_workbook(os.path.join(GENETICS_PATH,
                                           'genes_BMI_ob.xls'))
sheet1 = workbook.sheet_by_name('SNPs_from_literature')

SNPs_from_literature = []
n_rows_1 = 66
for row in range(n_rows_1):
    SNPs_from_literature.append(sheet1.cell(row, 0).value)
    SNPs_from_literature[row].replace("text:u", "").replace("'", "")

SNPs_from_literature = [str(snp) for snp in SNPs_from_literature]

# All SNPs extracted from genes associated with BMI (UCSC)
sheet2 = workbook.sheet_by_name('SNPs_from_BMI_genes')

SNPs_from_BMI_genes = []
n_rows_2 = 1615
for row in range(n_rows_2):
    SNPs_from_BMI_genes.append(sheet2.cell(row, 0).value)
    SNPs_from_BMI_genes[row].replace("text:u", "").replace("'", "")

SNPs_from_BMI_genes = [str(snp) for snp in SNPs_from_BMI_genes]

# Intersection of all SNPs extracted from BMI-associated genes and candidate
# SNPs specifically studied to have a link with BMI
SNPs_of_interest = np.intersect1d(SNPs_from_literature,
                                  SNPs_from_BMI_genes)

# SNPs distribution among IMAGEN subjects
BMI_SNPs_intercept_file_path = os.path.join(GENETICS_PATH,
                                            'BMI_SNPs_intercept.txt')

with open(BMI_SNPs_intercept_file_path, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
    for i, SNP in enumerate(SNPs_of_interest):
        spamwriter.writerow([SNP])

# Read genotypage data as a dataframe
SNPs_demographics = pd.io.parsers.read_csv(os.path.join(DATA_PATH,
                                    'BMI_associated_SNPs_demographics.csv'),
                                           index_col=0)

# Select only SNPs of interest to see if they are signigicantly represented
# throughout the population
SNPs_demographics = SNPs_demographics[SNPs_of_interest]

# Count number of heterozygotes, dominant and recessives homozygotes among
# subjects for whom we have both neuroimaging and genetic data
for i, j in enumerate(SNPs_of_interest):
    dominant_homoz = sum(SNPs_demographics[j] == 0)
    recessive_homoz = sum(SNPs_demographics[j] == 2)
    heterozygotes = sum(SNPs_demographics[j] == 1)
    print 'For SNP', j, 'there are:'
    print dominant_homoz, 'dominant homozygotes,'
    print recessive_homoz, 'recessive homozygotes,'
    print heterozygotes, 'heterozygotes.'


# SNPs distribution among IMAGEN subjects
SNPs_distribution_file_path = os.path.join(GENETICS_PATH,
                                           'SNPs_distribution.txt')

with open(SNPs_distribution_file_path, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')

    for i, j in enumerate(SNPs_of_interest):
        dominant_homoz = sum(SNPs_demographics[j] == 0)
        recessive_homoz = sum(SNPs_demographics[j] == 2)
        heterozygotes = sum(SNPs_demographics[j] == 1)

        spamwriter.writerow(['For SNP'] + [j] + ['there are:']
                            + [dominant_homoz] + ['dominant homozygotes,']
                            + [recessive_homoz] + ['recessive homozygotes,']
                            + [heterozygotes] + ['heterozygotes.'])