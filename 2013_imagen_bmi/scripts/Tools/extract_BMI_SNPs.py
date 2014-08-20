# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 13:43:10 2014

@author: vf140245

This script returns the list of all SNPs extracted from genes known to be
associated to BMI.

OUTPUT: csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/BMI_SNPs_names_list.csv"
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join('/home/vf140245', 'gits', 'mycaps/nsap/caps'))
from genim.genibabel import connect, load_from_genes


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')

# Main genes associated to BMI according to the literature
# Qualified name eGene NCBI
bmi_gene_list = ['BDNF', 'CADM2', 'COL4A3BP', 'ETV5', 'FAIM2', 'FANCL',
                 'FTO', 'GIPR', 'GNPDA2', 'GPRC5B', 'HMGCR', 'KCTD15',
                 'LMX1B', 'LRP1B', 'LINGO2', 'MAP2K5', 'MC4R', 'MTCH2',
                 'MTIF3', 'NEGR1', 'NPC1', 'NRXN3', 'NTRK2', 'NUDT3', 'POC5',
                 'POMC', 'PRKD1', 'PRL', 'PTBP2', 'PTER', 'QPCTL', 'RPL27A',
                 'SEC16B', 'SH2B1', 'SLC39A8', 'SREBF2', 'TFAP2B', 'TMEM160',
                 'TMEM18', 'TNNI3K', 'TOMM40', 'ZNF608']

if __name__ == "__main__":
    bioresDB = connect(server='mart.cea.fr', user='admin', passwd='alpine')

    snps_dict, void_gene, df = load_from_genes(bmi_gene_list,
                                               study='IMAGEN',
                                               bioresDB=bioresDB)
    # Pathnames
    BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
    DATA_PATH = os.path.join(BASE_PATH, 'data')

    # Write results in a single txt file for all genes

    # SNPs considered: SNPs from genes known to be associated to BMI
    BMI_SNPs = df.columns

    # SNPs considered are stored as a dataframe pandas and this list is saved
    # as a csv file for further use
    BMI_SNPs_names_list = pd.DataFrame.to_csv(pd.DataFrame(BMI_SNPs),
                                              os.path.join(DATA_PATH,
                                                  'BMI_SNPs_names_list.csv'),
                                              header=False,
                                              index=False)

#    #examples
#    a = df.loc[[u'000037509984', u'000044836688', u'000063400084'], :].values
#    b = df.loc[[u'000037509984', u'000063400084', u'000044836688'], :].values
#    subj_list = df.index    # subjects
#    snp_list = df.columns   # snps