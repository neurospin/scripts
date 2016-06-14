"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import optparse
import re, glob, os


## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_profil/'
basename = 'morpho_'
left_STAP = 'S.T.s._left.dat'
right_STAP = 'S.T.s._right.dat'

columns = ['geodesicDepthMax']
columns_f = ['FID', 'IID']+columns
df_right0 = pd.DataFrame()
df_left0 = pd.DataFrame()
for j in range(10):
    df_right = pd.read_csv(DIRECTORY_STAP+basename+str(j)+left_STAP, delim_whitespace=True)
    
    df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
    df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
    df_right.index = df_right['IID']
    if j == 0:
        df_right0['FID'] = df_right['FID']
        df_right0['IID'] = df_right['IID']
        df_right0.index = df_right0['IID']
        df_right0['geodesicDepthMax'] = np.zeros(len(df_right0.index))
    df_right0['geodesicDepthMax'+str(j)] = df_right['geodesicDepthMax']
    df_right0['geodesicDepthMax'] += df_right['geodesicDepthMax']
    df_left = pd.read_csv(DIRECTORY_STAP+basename+str(j)+left_STAP, delim_whitespace=True)
    df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
    df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
    df_left.index = df_left['IID']
    if j == 0:
        df_left0['FID'] = df_left['FID']
        df_left0['IID'] = df_left['IID']
        df_left0.index = df_left0['IID']
        df_left0['geodesicDepthMax'] = np.zeros(len(df_left0.index))
    df_left0['geodesicDepthMax'+str(j)] = df_left['geodesicDepthMax']
    df_left0['geodesicDepthMax'] += df_left['geodesicDepthMax']  
#df_left0['geodesicDepthMax'] = df_left0['geodesicDepthMax']/10
