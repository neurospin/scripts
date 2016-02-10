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
columns_qcovar = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'ICV']
columns_covar = [u'IID',u'FID', u'Centres_Berlin',u'Centres_Dresden',
                u'Centres_Dublin', u'Centres_Hamburg',u'Centres_London',
                u'Centres_Mannheim', u'Centres_Nottingham',u'SNPSEX']

gcta_qcovar = '/neurospin/brainomics/imagen_central/clean_covar/covar_5PCA_ICV_GCTA.qcov'
gcta_covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit_GCTA.cov'


df = pd.read_csv(megha_covar, delim_whitespace=True, header=None)
df.columns = columns_names
df['IID'] = ['%012d' % int(i) for i in df['IID']]
df['FID'] = ['%012d' % int(i) for i in df['FID']]
df_covar = df[columns_covar]
df_qcovar = df[columns_qcovar]
df_covar.to_csv(gcta_covar, sep= '	',  header=False, index=False)
df_qcovar.to_csv(gcta_qcovar, sep= '	',  header=False, index=False)
