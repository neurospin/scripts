"""
@author yl247234 
"""

import pandas as pd
import pheno
import numpy as np
import optparse
import re, glob, os

## INPUTS ##
DIRECTORY_PHENO = '/neurospin/brainomics/2016_sulcal_depth/isomap/imagen_central_sulcus_isomap/'
filename = 'isoSTSImagenDim5.txt'#'isoMDSCSSylk10d3_3_distmin.txt'#isoMDSCSSylk10d1distmin.txt'
## OUTPUT ##
OUT_DIRECTORY = '/neurospin/brainomics/2016_sulcal_depth/isomap/'
OUTPUT_FILE = 'left_right_asym-STS.phe'


df = pd.read_csv(DIRECTORY_PHENO+filename, delim_whitespace=True)
right_index = []
left_index = []
right_values = []
left_values = []
count = 0
for ind in df.index:
    if 'R' in ind:
        right_index.append(ind[len(ind)-12:len(ind)])
        right_values.append(df.loc[ind][0])
    elif 'L' in ind:
        left_index.append(ind[len(ind)-12:len(ind)])
        left_values.append(df.loc[ind][0])
    else:
        count += 1
    

df_right = pd.DataFrame({'IID' : np.asarray(right_index),
                         'FID' : np.asarray(right_index),
                     'sts_right' : np.asarray(right_values)})
df_right.index = df_right['IID']
df_right.to_csv(OUT_DIRECTORY+'right_V0-STS.phe', sep= '\t',  header=True, index=False)

df_left = pd.DataFrame({'IID' : np.asarray(left_index),
                        'FID' : np.asarray(left_index),
                        'sts_left' : np.asarray(left_values)})
df_left.index = df_left['IID']
df_left.to_csv(OUT_DIRECTORY+'left_V0-STS.phe', sep= '\t',  header=True, index=False)


df = df_right
df[df_left.columns] = df_left[df_left.columns]
df = df.dropna(axis=0)
df['asym'] = ((df['sts_left']-df['sts_right'])/(df['sts_left']+df['sts_right']))
df.to_csv(OUT_DIRECTORY+OUTPUT_FILE, sep= '\t',  header=True, index=False)
