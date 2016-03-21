"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os

## INPUTS ##
DIRECTORY_PHENO = '/neurospin/brainomics/2016_hippo_malrot/data/'
filename = 'IMAGEN_allNot_cleaned.xlsx'
#columns = ['FID', 'IID', 'Sci_L_thresh', 'Sci_R_thresh','SCi_L', 'SCi_R','C0_L', 'C0_R']
columns0 = ['FID', 'IID', 'Sci_L_thresh', 'Sci_R_thresh', 'SCi_L', 'SCi_R']

## OUTPUT ##
OUT_DIRECTORY = '/neurospin/brainomics/2016_hippo_malrot/pheno_update/'
OUTPUT_FILE = 'hippo_IHI.phe'


df = pd.read_excel(DIRECTORY_PHENO+filename)
df['IID'] = df['Subject']
df['FID'] = df['Subject']
df1 = df[columns0]
df = df1.dropna()
SCi_L = np.log(np.asarray(df['SCi_L']+1).tolist())
SCi_R = np.log(np.asarray(df['SCi_R']+1).tolist())
SCi_L_R = np.log(np.asarray(df['SCi_L']+df['SCi_R']+1).tolist())
df['SCi_L'] = SCi_L
df['SCi_R'] = SCi_R
df['SCi_L_R'] = SCi_L_R
"""for j in range(len(df['C0_L'])):
    if df['C0_L'][j] == 'Y':
        df['C0_L'][j] = 1
    elif df['C0_L'][j] == 'N':
        df['C0_L'][j] = 2
    else:
        df['C0_L'][j] = np.nan
for j in range(len(df['C0_R'])):
    if df['C0_R'][j] == 'Y':
        df['C0_R'][j] = 1
    elif df['C0_R'][j] == 'N':
        df['C0_R'][j] = 2
    else:
        df['C0_R'][j] = np.nan
df = df.dropna()"""
df['IID'] = ['%012d' % int(i) for i in df['IID']]
df['FID'] = ['%012d' % int(i) for i in df['FID']]
df['Sci_L_thresh'] = [int(i)+1 for i in df['Sci_L_thresh']]
df['Sci_R_thresh'] = [int(i)+1 for i in df['Sci_R_thresh']]
df.index = df['IID']

df.to_csv(OUT_DIRECTORY+OUTPUT_FILE, sep= '\t',  header=True, index=False)



import matplotlib.pyplot as plt



number_bins = 100


n, bins, patches = plt.hist(SCi_L, number_bins)
plt.figure()
n, bins, patches = plt.hist(SCi_R, number_bins)
plt.figure()
n, bins, patches = plt.hist(SCi_L_R, number_bins)
plt.show()
