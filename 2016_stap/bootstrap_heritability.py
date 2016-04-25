"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math, random


## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_stap/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=None)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'SNPSEX', u'ICV'] 
df_covar['IID'] = ['%012d' % int(i) for i in df_covar['IID']]
df_covar.index = df_covar['IID']

## OUTPUT ##
pheno_name = 'STAP'
extension = '.phe'


## Preprocessing ##
columns = ['geodesicDepthMax']
columns_f = ['FID', 'IID']+columns
df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]
df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]
df = pd.DataFrame()
df['asym_max'] =2*(df_right['geodesicDepthMax']-df_left['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])
df['IID'] = df.index
df['FID'] = df.index
df['right_depth_max'] = df_right['geodesicDepthMax']
df['left_depth_max'] = df_left['geodesicDepthMax']
df = df[['FID', 'IID', 'asym_max', 'left_depth_max', 'right_depth_max']] 
df = df.dropna() # 2080 subjects (no NaN)
df = df.loc[df_covar.index] # 1763 subjects
df = df.dropna() # 1754 subjects ..

for j in range(7):
    n_subjects = 1100+j*100
    DIRECTORY_PHENO = DIRECTORY_STAP+'Phenotypes_bootstrap_100samples/Subjects'+str(n_subjects)+'/'
    if not os.path.exists(DIRECTORY_PHENO):
        os.makedirs(DIRECTORY_PHENO)
    for i in range(100):
        #s_ids = random.sample(set(df.index)-set(['000060281382', '000000297685', '000007867512', '000009715842', '000010400245', '000011104036', '000012119704', '000012645188']), n_subjects) 
        s_ids = random.sample(set(df.index), n_subjects)
        df_temp = df.loc[s_ids]
        df_temp.to_csv(DIRECTORY_PHENO+pheno_name+str(i)+extension, sep= '\t',  header=True, index=False)

