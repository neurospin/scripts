"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os, json

## INPUTS ##
DIRECTORY_GENES = '/neurospin/brainomics/2016_hippo_malrot/data/'
filename = 'Genes_Hippocampus.txt'
## OUTPUTS ##
out_snps = '/neurospin/brainomics/2016_hippo_malrot/results_snps/IHI_genotypes.csv'

# For measured genotypes
from genibabel import imagen_genotype_measure


# Consider subjects for who we have neuroimaging and genetic data
# To fix genibabel should offer a iid function -direct request to server
login = json.load(open(os.environ['KEYPASS']))['login']
password = json.load(open(os.environ['KEYPASS']))['passwd']  

#Example 1: requesting genotypes 
df = pd.read_csv('/neurospin/brainomics/2016_hippo_malrot/data/Genes_Hippocampus.csv', header=None)
IHI_gene_names =  np.concatenate(np.asarray(df), axis=0)
IHI_gene_names = IHI_gene_names.tolist()
# for measured data
ihi_genotypes = imagen_genotype_measure(login,
                                        password,
                                        gene_names=IHI_gene_names)
# export data to CSV
ihi_genotypes.csv_export(out_snps)
