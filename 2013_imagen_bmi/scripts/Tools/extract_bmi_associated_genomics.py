# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:03:24 2014

@author: hl237680

Copyrignt : CEA NeuroSpin - 2014

This script generates a .csv file containing the genotype of IMAGEN subjects
for whom we have both neuroimaging and genetic data regarding the SNPs from
genes robustly associated to BMI.

INPUT:
- Genotype data from the IMAGEN study:
    "/neurospin/brainomics/2013_imagen_bmi/2012_imagen_shfj/genetics/
    qc_sub_qc_gen_all_snps_common_autosome"
- List of SNPs from genes found in the literature to be strongly associated
  to BMI:
    "/neurospin/brainomics/2013_imagen_bmi/data/BMI_SNPs_names_list.csv"
- List of the 1.265 subjects for whom we have both neuroimaging and genetic
  data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT: .csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/
    BMI_associated_SNPs_demographics.csv"

"""

import numpy as np
import os
import pandas as pd
import plinkio as ig


## Pathnames ##
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')


## Functions ##

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


# SNPs from genes highly associated to BMI
BMI_SNPs = np.genfromtxt(os.path.join(DATA_PATH, 'BMI_SNPs_names_list.csv'),
                         dtype=None)

# Genotype data (IMAGEN study)
gfn = os.path.join('/neurospin/brainomics',
                   '2012_imagen_shfj',
                   'genetics',
                   'qc_sub_qc_gen_all_snps_common_autosome')

genotype = ig.Genotype(gfn)

# Build an array that keeps only SNPs of interest among the SNPs referenced
# in the qc_sub_qc_gen_all_snps_common_autosome file
SNPs = genotype.snpGenotypeByName(BMI_SNPs.tolist())

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
                           columns=BMI_SNPs)
SNPs_df = SNPs_IMAGEN.loc[subjects_id, :]

# Write SNPs of interest in a .csv file
SNPs_df.to_csv(os.path.join(DATA_PATH, 'BMI_associated_SNPs_demographics.csv'))
print "Saved .csv containing BMI associated SNPs within the IMAGEN population."