# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 18:32:17 2014

@author: hl237680

This script returns the list of all SNPs extracted from genes known to be
associated to BMI that are present on mart CW database, and a dataframe
containing genotype associated measures.

INPUT:
- mart database for info on SNPs measures for the IMAGEN cohort
- subjects id of subjects for whom we have both neuroimaging and genetic data:
    "/neurospin/brainomics/2013_imagen_bmi/data/subjects_id.csv"

OUTPUT:
- list of SNPs included in genes known to be associated to the BMI that are
  present on the mart database:
    "/neurospin/brainomics/2013_imagen_bmi/data/BMI_SNPs_names_list.csv"
- csv file containing SNPs' measures for subjects for whom we have both
  neuroimaging and genetic data:
    "/neurospin/brainomics/2013_imagen_bmi/data/SNPs_all.csv"

BEWARE!!!
The path to get genetic data online (mart) has changed. Here, it is the old
script to bypass CW database problems before having checked the consistency
of the database by unit tests.
Besides, genetic data have to be imputed.
    
    => See 02_IMAGEN1265_SNPs_measurements.py

"""


import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join('/home/vf140245', 'gits', 'mycaps/nsap/caps'))
from genim.genibabel import connect, load_from_genes


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')


# Main genes associated to BMI according to the literature
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

    # SNPs considered: SNPs from genes known to be associated to BMI
    BMI_SNPs = df.columns

    # SNPs considered are stored as a dataframe pandas and this list is saved
    # as a csv file for further use, i.e. to check their correlation to
    # obesity with Plink
    BMI_SNPs_names_list = pd.DataFrame.to_csv(pd.DataFrame(BMI_SNPs),
                                              os.path.join(DATA_PATH,
                                                  'BMI_SNPs_names_list.csv'),
                                              header=False,
                                              index=False)

    # Get the ordered list of subjects ID
    subjects_id = np.genfromtxt(os.path.join(DATA_PATH, 'subjects_id.csv'),
                                dtype=None, delimiter=',', skip_header=1)

    # Conversion to unicode with 12 positions
    subjects_id = [unicode('%012d' % i) for i in subjects_id]

    # SNPs_IMAGEN: dataframe with index giving subjects ID in the right order
    # and columns the SNPs considered
    SNPs_IMAGEN = df.loc[subjects_id, :]

    # Write all SNPs' measures for subjects for whom we have both neuroimaging
    # and genetic data in a .csv file
    SNPs_IMAGEN.to_csv(os.path.join(DATA_PATH, 'SNPs_all.csv'))
    print "SNPs_IMAGEN saved to SNPs_all.csv"