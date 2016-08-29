"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os, json

## INPUTS ##
DIRECTORY_SNPS = '/neurospin/brainomics/2016_hippo_malrot/PLINK_output/parse_output_update/'
pheno_names = ['Sci_L_thresh', 'Sci_R_thresh', 'SCi_L', 'SCi_R', 'SCi_L_R']
case = ''


for pheno_name in pheno_names:
    print '\n'
    print "PHENOTYPE: " +pheno_name
    
    ## OUTPUT ##
    out_genes = '/neurospin/brainomics/2016_sulcal_depth/genibabel_output/'+pheno_name+'_genes.csv'
    
    pval_sel = DIRECTORY_SNPS+pheno_name+case+'.sel5'

    if os.path.isfile(pval_sel):
        pval = pd.read_csv(pval_sel, sep='\t')
    snps = np.asarray(pval['SNP'])
    pvalues = np.asarray(pval['P'])
    snps = snps.tolist()



    # For measured genotypes
    """from genibabel import imagen_genotype_measure


    # Consider subjects for who we have neuroimaging and genetic data
    # To fix genibabel should offer a iid function -direct request to server
    login = json.load(open(os.environ['KEYPASS']))['login']
    password = json.load(open(os.environ['KEYPASS']))['passwd']  


    # for measured data
    STAP_genotypes = imagen_genotype_measure(login,
                                            password,
                                            snp_ids=snps)
    # export data to CSV
    STAP_genotypes.csv_export(out_genes)"""

    with open(out_genes) as f:
        lis=[line.split(';') for line in f]
    genes = lis[1][2:]
    genes[len(genes)-1]=genes[len(genes)-1][:len(genes[len(genes)-1])-2]
    corresponding_snps = lis[3][2:]
    corresponding_snps[len(corresponding_snps)-1]=corresponding_snps[len(corresponding_snps)-1][:len(corresponding_snps[len(corresponding_snps)-1])-2]

    df = pd.DataFrame({'corresponding_snps' : np.asarray(corresponding_snps),
                       'genes' : np.asarray(genes),
                       'pval': np.asarray(pvalues)})

    df.index = df['corresponding_snps']
    print df
