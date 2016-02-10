"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os

### INPUTS ###
megha_covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov'
columns_names = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                 u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                 u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                 u'SNPSEX', u'ICV'] 
### OUTPUTS ###

plink_covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_PLINK.cov'


df = pd.read_csv(megha_covar, delim_whitespace=True, header=None)
df.columns = columns_names
df['IID'] = ['%012d' % int(i) for i in df['IID']]
df['FID'] = ['%012d' % int(i) for i in df['FID']]
df.to_csv(plink_covar, sep= '	',  header=True, index=False)
