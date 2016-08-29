"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os, json
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26
## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'
## OUTPUT ##
left_pheno = 'left_STAP_hull'
right_pheno = 'right_STAP_hull'
asym_pheno = 'asym_STAP_hull'
pheno_name = 'STAP_hull_cap'
extension = '.phe'
DIRECTORY_PHENO = DIRECTORY_STAP+'new_pheno/8th_filter/'


columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage', 'hullJunctionsLength']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]
df_right['plisDePassage'] = p 
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right0 = df_right[columns_f]
df_right0 = df_right0.loc[df_right0['hullJunctionsLength']>40]

df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]
df_left['plisDePassage'] = p 
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left0 = df_left[columns_f]
df_left0 = df_left0.loc[df_left0['hullJunctionsLength']>40]

# hull from 54 to 100
for cap in range(54,102,2):
    df_right = df_right0.loc[df_right0['hullJunctionsLength']<cap]
    df_left = df_left0.loc[df_left0['hullJunctionsLength']<cap]
    df_left = df_left[['FID', 'IID','geodesicDepthMax']]
    df_right = df_right[['FID', 'IID','geodesicDepthMax']]
    df_right = df_right.loc[df_left.index]
    df_right = df_right.dropna()
    df_left = df_left.loc[df_right.index]
    df_left = df_left.dropna()
    df = pd.DataFrame()
    df['asym_max'] =(df_left['geodesicDepthMax']-df_right['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])
    df['IID'] = df.index
    df['FID'] = df.index
    df['right_depth_max'] = df_right['geodesicDepthMax']
    df['left_depth_max'] = df_left['geodesicDepthMax']
    df = df[['FID', 'IID', 'asym_max', 'left_depth_max', 'right_depth_max']]
    df.to_csv(DIRECTORY_PHENO+pheno_name+str(cap)+extension, sep= '\t',  header=True, index=False)


