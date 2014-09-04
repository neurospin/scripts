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

OUTPUT:
- SNPs_of_interest: intercept of both lists of SNPs
- SNPs distribution among IMAGEN subjects:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    SNPs_distribution.txt"

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

# Intersection of all SNPs extracted from BMI-associated genes and candidate
# SNPs specifically studied to have a link with BMI
SNPs_of_interest = np.intersect1d(SNPs_from_literature,
                                  SNPs_from_BMI_genes)

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
SNPs_distribution_file_path = os.path.join(OUTPUT_DIR, 'SNPs_distribution.txt')

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