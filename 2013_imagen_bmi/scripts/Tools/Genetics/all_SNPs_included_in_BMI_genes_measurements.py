# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 09:46:04 2014

@author: hl237680

Copyrignt : CEA NeuroSpin - 2014

This script generates a .csv file containing the genotype of IMAGEN subjects
for whom we have both neuroimaging and genetic data regarding SNPs of interest
at the intersection between all SNPs included in BMI-associated genes and
SNPs read by the Illumina platform on IMAGEN subjects.

NB: Difference with the script IMAGEN1265_SNPs_measurements.py is that here,
    we do not only focus on SNPs referenced in the literature as associated
    to the BMI.

INPUT:
- Genotype data from the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/2012_imagen_shfj/genetics/
    qc_sub_qc_gen_all_snps_common_autosome"
- List of all SNPs included in BMI-associated genes:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Plink/Sulci_SNPs/
    all_SNPs_within_BMI_associated_genes.snp"
- List of the 1.265 subjects for whom we have both neuroimaging and genetic
  data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT:
- Intersection between all SNPs included in BMI-associated genes and SNPs
  read by the Illumina platform for the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    all_Illumina_SNPs_from_BMI_genes.csv"
- Genetic measures on SNPs of interest
    "/neurospin/brainomics/2013_imagen_bmi/data/
    BMI_associated_all_SNPs_measures.csv"

"""

import os
import numpy as np
import csv
import pandas as pd
import plinkio as ig


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')
PLINK_PATH = os.path.join(GENETICS_PATH, 'Plink')
SULCI_SNPs_PLINK_PATH = os.path.join(PLINK_PATH, 'Sulci_SNPs')


# Functions
def impute_data_by_med(data, verbose=0, nan_symbol=128):
    """ This function cut/pasted from B DaMota (genim-stat)
    """
    eps2 = 1e-17
    asNan = (data == nan_symbol)
    if verbose:
        print ('impute %d data for a total of %d' %
        (asNan[asNan].size, data.shape[0] * data.shape[1]))
    med = np.array([int(np.median(data[:, i]))
                    for i in range(0, data.shape[1])])
    if med[med > 2].size > 0:
        print 'med == %s :' % str(nan_symbol), med[med > 2].size
        print 'med shape :', med.shape
        print 'shape of repetition', data.shape[0]
    else:
        med = np.array([np.median(data[:, i])
                       for i in range(0, data.shape[1])])
        med[med == 0] = eps2
    med_all = np.repeat(med, data.shape[0]).reshape((-1, data.shape[0])).T
    data[asNan] = med_all[asNan]

    return data


if __name__ == "__main__":

    # All SNPs included in BMI-associated genes
    all_BMI_SNPs = np.genfromtxt(os.path.join(SULCI_SNPs_PLINK_PATH,
                                  'all_SNPs_within_BMI_associated_genes.snp'),
                                 dtype=None,
                                 delimiter=',',
                                 skip_header=0).tolist()

    # Genotype data (IMAGEN study)
    gfn = os.path.join('/neurospin/brainomics',
                       '2012_imagen_shfj',
                       'genetics',
                       'qc_sub_qc_gen_all_snps_common_autosome')

    genotype = ig.Genotype(gfn)

    # Build an array - later converted to a list - that keeps only SNPs of
    # interest among the SNPs referenced in the
    # qc_sub_qc_gen_all_snps_common_autosome file
    snp_from_BMI_genes_in_study = np.intersect1d(all_BMI_SNPs,
                                         genotype.snpList().tolist()).tolist()

    # Save list of SNPs at the intersection between BMI-associated SNPs
    # referenced in the literature and SNPs explored by the Illumina platform
    snp_from_BMI_genes_in_study_file = os.path.join(GENETICS_PATH,
                                        'all_Illumina_SNPs_from_BMI_genes.csv')

    with open(snp_from_BMI_genes_in_study_file, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
        for i, SNP in enumerate(snp_from_BMI_genes_in_study):
            spamwriter.writerow([SNP])

    # Genotype of IMAGEN subjects for SNPs snp_from_BMI_genes_in_study
    SNPs = genotype.snpGenotypeByName(snp_from_BMI_genes_in_study)

    # Remove potential missing data
    SNPs = impute_data_by_med(SNPs)

    # Subjects ID for whom we have both neuroimaging and genetic data
    subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                                dtype=None,
                                delimiter=',',
                                skip_header=1)

    # Conversion to unicode with 12 positions
    subjects_id = [unicode('%012d' % i) for i in subjects_id]

    # SNPs_df: dataframe with index giving subjects ID in the right order
    # and columns the SNPs considered
    SNPs_IMAGEN = pd.DataFrame(SNPs,
                               index=genotype.assayIID(),
                               columns=snp_from_BMI_genes_in_study)
    SNPs_df = SNPs_IMAGEN.loc[subjects_id, :]

    # Write genetic measures for SNPs of interest in a .csv file
    SNPs_df.to_csv(os.path.join(DATA_PATH,
                                'SNPs_from_BMI-associated_genes_measures.csv'))
    print ".csv containing measures for all SNPs included in BMI-associated genes at the intersection with Illumina read SNPs."