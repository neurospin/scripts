# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 18:42:00 2014

@author: hl237680

Extract SNPs of interest (Graff, Nature, 2012) from the IMAGEN cohort,
put subjects ID in the right order,
get a dataframe where index=subjects_id and columns=SNPs,
convert dataframe into a numpy array.
"""

import os, sys
import numpy as np

sys.path.append(os.path.join('/home/vf140245', 'gits', 'mycaps/nsap/caps'))
from genim.genibabel import connect, load_from_genes


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
BMI_FILE = os.path.join(DATA_PATH, 'BMI.csv')

# Shared data
BASE_SHARED_DIR = "/neurospin/tmp/brainomics/"


# List from Graff paper Nature 2012
# qualified name eGene NCBI
bmi_gene_list = ['SEC16B', 'TNNI3K', 'PTBP2', 'NEGR1', 'LYPLAL1', 'LZTR2',
                 'LRP1B', 'TMEM18', 'POMC', 'FANCL', 'CADM2', 'SLC39A8',
                 'FLJ3577', 'HMGCR', 'NCR3', 'AIF1', 'BAT2', 'NUDT3',
                 'TFAP2B', 'MSRA', 'LRRN6C', 'LMX1B', 'BDNF', 'MTCH2',
                 'RPL27A', 'TUB', 'FAIM2', 'MTIF3', 'NRXN3', 'PRKD1',
                 'MAP2K5', 'GPRC5B', 'ADCY9', 'SH2B1', 'APOB48', 'FTO',
                 'MC4R', 'QPCTL', 'KCTD15', 'TMEM160', 'PRL', 'PTER', 'MAF',
                 'NPC1']

if __name__ == "__main__":
    bioresDB = connect(server='mart.cea.fr', user='admin', passwd='alpine')
    snps_dict, void_gene, df = load_from_genes(bmi_gene_list,
                                               study='IMAGEN', 
                                               bioresDB = bioresDB)
    
    # Get the ordered list of subjects ID
    subjects_id = np.genfromtxt(os.path.join(DATA_PATH, "subjects_id.csv"), dtype=None, delimiter=',', skip_header=1)    
    subjects_id = [unicode("%012d"%i)for i in subjects_id]  # Conversion to unicode with 12 positions
    
    # SNPs_IMAGEN: dataframe with index giving subjects ID in the right order and columns the SNPs considered (from Graff, Nature, 2012)
    SNPs_IMAGEN = df.loc[subjects_id,:]
    #subj_list = SNPs_IMAGEN.index   #subjects
    snp_list = SNPs_IMAGEN.columns  #snps
    # Write SNPs in column a .txt file to check their correlation to obesity with Plinkio
    with open(os.path.join(DATA_PATH, "snp_list.txt"), "w") as f:
        for snp in snp_list:
            print >> f, snp
    
    # Write SNPs for all subjects in a .csv file
    SNPs_IMAGEN.to_csv(os.path.join(DATA_PATH, "SNPs_hl.csv"))
    print "SNPs_IMAGEN saved to SNPs_hl.csv"