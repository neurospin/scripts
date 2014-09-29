# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:03:24 2014

@author: hl237680

Copyrignt : CEA NeuroSpin - 2014

This script generates a .csv file containing the genotype of IMAGEN subjects
for whom we have both neuroimaging and genetic data regarding SNPs of interest
at the intersection between SNPs referenced in the literature as from genes
robustly associated to BMI and SNPs read by the Illumina platform on IMAGEN
subjects.

INPUT:
- Genotype data from the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/2012_imagen_shfj/genetics/
    qc_sub_qc_gen_all_snps_common_autosome"
- SNPs referenced in the literature as strongly associated to the BMI
  (included in or near to BMI-associated genes):
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    genes_BMI_ob.xls"
- List of the 1.265 subjects for whom we have both neuroimaging and genetic
  data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT:
- List of SNPs referenced in the literature as strongly associated to the BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    BMI_SNPs_names_list.csv"
- List of SNPs at the intersection between BMI-associated SNPs referenced in
  the literature and SNPs read by the Illumina platform:
    "/neurospin/brainomics/2013_imagen_bmi/data/genetics/
    snp_from_liter_in_study.csv"
- Genetic measures on SNPs of interest
    "/neurospin/brainomics/2013_imagen_bmi/data/
    BMI_associated_SNPs_measures.csv"

"""

import os
import numpy as np
import xlrd
import csv
import pandas as pd
import plinkio as ig


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
GENETICS_PATH = os.path.join(DATA_PATH, 'genetics')


# Functions

#def extract(genotype, snps_dict):
#    """ from a genotype instance provide various helpers
#    """
#    void_gene = [i for i in snps_dict if len(snps_dict[i]) == 0]
#    _ = [snps_dict.pop(i) for i in void_gene]
#    col = []
#    _ = [col.extend(snps_dict[i]) for i in snps_dict]
#    col = [str(i) for i in col]
#    data = genotype.snpGenotypeByName(col)
#    data = impute_data_by_med(data, verbose=True, nan_symbol=128)
#    row = genotype.assayIID()
#
#    return data, col, row, snps_dict, void_gene


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


## Build a .csv file containing all useful genetic data for the study of
## BMI on the IMAGEN population

#a = (open('/neurospin/brainomics/2013_imagen_bmi/data/genetics/blabla.annot').
#        read().split('\n')[:-1])
#a = [i.split() for i in a]
#a = [(i[3], i[7].split('|')[1]) for i in a]
#snp_dict = dict()
#gene = list(np.unique([i[1] for i in a]))
#
#for i in gene:
#    snp_dict[i] = []
#for i in a:
#    snp_dict[i[1]].append(i[0])
#
#gfn = os.path.join('/neurospin/brainomics',
#                   '2012_imagen_shfj',
#                   'genetics',
#                   'qc_sub_qc_gen_all_snps_common_autosome.bim')
#
#tmp = [i.split('\t')[1] for i in open(gfn).read().split('\n')[:-1]]
#universe = set(tmp)
#
#for i in snp_dict:
#    snp_dict[i] = set(snp_dict[i]).intersection(universe)
#
#snp_data, snp_data_columns, snp_data_rows, snp_dict, void_gene = \
#                                              extract(genotype, snp_dict)
#
#df = pd.DataFrame(snp_data,
#                  index=snp_data_rows,
#                  columns=snp_data_columns)


if __name__ == "__main__":

    # SNPs referenced in the literature as highly associated to the BMI
    workbook = xlrd.open_workbook(os.path.join(GENETICS_PATH,
                                               'genes_BMI_ob.xls'))
    sheet = workbook.sheet_by_name('SNPs_from_literature')

    SNPs_from_literature = []
    n_rows = 67
    for row in range(n_rows):
        SNPs_from_literature.append(sheet.cell(row, 0).value)
        SNPs_from_literature[row].replace("text:u", "").replace("'", "")

    SNPs_from_literature = [str(snp) for snp in SNPs_from_literature]

    # Save list of SNPs of interest
    BMI_SNPs_from_literature = os.path.join(GENETICS_PATH,
                                            'BMI_SNPs_names_list.csv')

    with open(BMI_SNPs_from_literature, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
        for i, SNP in enumerate(SNPs_from_literature):
            spamwriter.writerow([SNP])

    # Genotype data (IMAGEN study)
    gfn = os.path.join('/neurospin/brainomics',
                       '2012_imagen_shfj',
                       'genetics',
                       'qc_sub_qc_gen_all_snps_common_autosome')

    genotype = ig.Genotype(gfn)

    # Build an array - later converted to a list - that keeps only SNPs of
    # interest among the SNPs referenced in the
    # qc_sub_qc_gen_all_snps_common_autosome file
    snp_from_liter_in_study = np.intersect1d(SNPs_from_literature[1:],
                                    genotype.snpList().tolist()).tolist()

    # Save list of SNPs at the intersection between BMI-associated SNPs
    # referenced in the literature and SNPs explored by the Illumina platform
    snp_from_liter_in_study_file = os.path.join(GENETICS_PATH,
                                                'snp_from_liter_in_study.csv')

    with open(snp_from_liter_in_study_file, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar=' ')
        for i, SNP in enumerate(snp_from_liter_in_study):
            spamwriter.writerow([SNP])

    # Genotype of IMAGEN subjects for SNPs snp_from_liter_in_study
    SNPs = genotype.snpGenotypeByName(snp_from_liter_in_study)

    ## Check whether there are missing data
    ## dominant homozygote
    #sum(SNPs[:, 0] == 1)
    ## heterozygote
    #sum(SNPs[:, 0] == 0)
    ## two muted alleles
    #sum(SNPs[:, 0] == 2)
    ## missing data
    #sum(SNPs == 128)

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
                               columns=snp_from_liter_in_study)
    SNPs_df = SNPs_IMAGEN.loc[subjects_id, :]

    # Write genetic measures for SNPs of interest in a .csv file
    SNPs_df.to_csv(os.path.join(DATA_PATH,
                                'BMI_associated_SNPs_measures.csv'))
    print "Saved .csv containing BMI-associated SNPs measures within the IMAGEN population."