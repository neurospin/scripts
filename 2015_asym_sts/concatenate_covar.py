# -*- coding: utf-8 -*-
"""
Created on Fri Nov  13 15:04 2015

@author: yl247234
Copyrignt : CEA NeuroSpin - 2014
"""

import os, glob, re
import pheno 
import numpy as np
import pandas as pd

out = '/volatile/yann/imagen_central/covar/MEGHA_covar.cov'
path = '/volatile/yann/imagen_central/covar/'
df1 = pd.read_csv(path+'AgeIBS.qcovar', delim_whitespace=True, header=None, names=[u'IID', u'FID', 'Age', 'C1', 'C2', 'C3', 'C4'])

df1.index = df1[u'IID']
df2 = pd.read_csv(path+'covar_GenCit_MEGHA.cov', delim_whitespace=True,  header=None, names=['IID','FID', 'Gender_Female',	'City_BERLIN',	'City_DRESDEN',	'City_DUBLIN',	'City_HAMBURG',	'City_LONDON',	'City_MANNHEIM', 'City_NOTTINGHAM'])
df2.index = df2[u'IID']
df = df1
df[df2.columns]=df2[df2.columns]
for column in df.columns:
    df = df.loc[np.logical_not(np.isnan(df[column]))]

df['IID'] = ['%012d' % int(i) for i in df['IID']] 
df.index = df[u'IID']
df['FID'] = df['IID']
df.to_csv(out, sep= '\t', header=False, index=False)

path ='/volatile/yann/2015_asym_sts/PLINK_all_pheno0.02/'
out = path+'concatenated_pheno_without_asym.phe'
tol = 0.02
count = 0
for filename in glob.glob(os.path.join(path,'*tol'+str(tol)+'.phe')):
    if count == 0:
        df  = pd.read_csv(filename, delim_whitespace=True)
        df.index = df[u'IID']
        df0 = pd.DataFrame()
        df0['IID'] = df[u'IID']
        df0.index = df0[u'IID']
        for column in df.columns:
            if 'asym' not in column:
                df0[column] = df[column]
        count += 1
    else:
        df_temp  = pd.read_csv(filename, delim_whitespace=True)
        df_temp.index = df_temp[u'IID']
        for column in df_temp.columns:
            if 'asym' not in column and 'IID' not in column and 'FID' not in column:
                df0[column]=df_temp[column]
        
for column in df.columns:
    df0 = df0.loc[np.logical_not(np.isnan(df[column]))]

df0['IID'] = ['%012d' % int(i) for i in df0['IID']]
df0.index = df0['IID']
df0['FID'] = ['%012d' % int(i) for i in df0['FID']]
df0.to_csv(out, sep= '\t', header=True, index=False)
