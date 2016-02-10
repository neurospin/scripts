"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os, json

## INPUTS ##
DIRECTORY_SNPS = '/neurospin/brainomics/2016_hippo_malrot/results_snps/'
input_snps = DIRECTORY_SNPS+'IHI_genotypes.csv'
pheno_names = ['SCi_L', 'SCi_R','Sci_L_thresh', 'Sci_R_thresh','C0_L', 'C0_R']
WORKING_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/'

## OUTPUTS ##
"""Only printing for the moment """

with open(input_snps) as f:
    lis=[line.split(';') for line in f] 

genes = lis[1][2:]
genes[len(genes)-1]=genes[len(genes)-1][:len(genes[len(genes)-1])-2]
corresponding_snps = lis[3][2:]
corresponding_snps[len(corresponding_snps)-1]=corresponding_snps[len(corresponding_snps)-1][:len(corresponding_snps[len(corresponding_snps)-1])-2]


df = pd.DataFrame({'corresponding_snps' : np.asarray(corresponding_snps),
                   'genes' : np.asarray(genes)})

df.index = df['corresponding_snps']

for pheno_name in pheno_names:
    pval_sel = os.path.join(WORKING_DIRECTORY,
                            pheno_name +'_logistic_pruned.sel4')
    pval = pd.read_csv(pval_sel, sep='\t')
    tab = np.asarray(pval['SNP'])
    tab = tab.tolist()
    for snp in tab:
        if snp in df.index:
            print "Phenotype considered: " + pheno_name
            print "Gene associated: "+ df.loc[snp][1]
            print "Snp associated: " + df.loc[snp][0]
