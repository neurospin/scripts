"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os, json

WORKING_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/parse_output_update/'
df = pd.read_csv('/neurospin/brainomics/2016_hippo_malrot/data/snps_SI_study_12q14_12q24.csv', header=None)
df = pd.read_csv('/neurospin/brainomics/2016_hippo_malrot/data/snps_two_studies_hippocampus.csv', header=None)
snps_names =  np.concatenate(np.asarray(df), axis=0)
pheno_names = ['Sci_L_thresh', 'Sci_R_thresh', 'SCi_L', 'SCi_R', 'SCi_L_R']
case = ''
for pheno_name in pheno_names:
    pval_sel = os.path.join(WORKING_DIRECTORY, pheno_name +case+'.sel3')
    pval = pd.read_csv(pval_sel, sep='\t')
    tab = np.asarray(pval['SNP'])
    tab = tab.tolist()
    print set(snps_names).intersection(tab)
