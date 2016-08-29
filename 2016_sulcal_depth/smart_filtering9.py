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
DIRECTORY_PHENO = DIRECTORY_STAP+'new_pheno/9th_filter/'

### DATA FOR ALLOMETRY REGULARIZATION ####
brainvisa_icv = '/neurospin/imagen/workspace/cati/morphometry/volumes/BL_morphologist_tissues_volumes.csv'
df21 = pd.read_csv(brainvisa_icv, sep=';')
df21['IID'] = ['%012d' % int(i) for i in df21['subject'] ]
df21.index = df21['IID']

covar_path = '/neurospin/brainomics/imagen_central/covar/'
df1 = pd.read_csv(covar_path+'aseg_stats_volume_BL.csv')
df1 = df1[['Measure:volume', 'EstimatedTotalIntraCranialVol']]
df1.columns = ['IID', 'ICV']
df1.index = df1['IID']
df1 = df1.sort_index(axis=0)
df1['IID'] = ['%012d' % int(i) for i in df1['IID']]
df1.index = df1['IID']

df2 = df21.loc[df1.index]
df3 = df1.loc[df21.index]
df2 = df2.dropna()
df3 = df3.dropna()

eTIV_Bv = np.asarray(df2['eTIV'])
eTIV_Freesurfer = np.asarray(df3['ICV'])
allometry_coeffs = {}
### END OF DATA FOR ALLOMETRY REGULARIZATION ####

columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage', 'hullJunctionsLength']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]
df_right['plisDePassage'] = p 
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]
### ALLOMETRY REGULARIZATION ####
df_right = df_right.loc[df2.index]
df_right = df_right.dropna()
x = np.log(eTIV_Bv)
y = np.log(df_right['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMean'] = p[0]
df_right['geodesicDepthMean'] = df_right['geodesicDepthMean']/np.power(eTIV_Bv,p[0])
y = np.log(df_right['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMax'] = p[0]
df_right['geodesicDepthMax'] = df_right['geodesicDepthMax']/np.power(eTIV_Bv,p[0])
### END OF ALLOMETRY REGULARIZATION ####
df_right0 = df_right.loc[df_right['hullJunctionsLength']>40]

df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]
df_left['plisDePassage'] = p 
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]
### ALLOMETRY REGULARIZATION ####
df_left = df_left.loc[df2.index]
df_left = df_left.dropna()
x = np.log(eTIV_Bv)
y = np.log(df_left['geodesicDepthMean'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMean'] = p[0]
df_left['geodesicDepthMean'] = df_left['geodesicDepthMean']/np.power(eTIV_Bv,p[0])
y = np.log(df_left['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMax'] = p[0]
df_left['geodesicDepthMax'] = df_left['geodesicDepthMax']/np.power(eTIV_Bv,p[0])
### END OF ALLOMETRY REGULARIZATION ####
df_left0 = df_left.loc[df_left['hullJunctionsLength']>40]



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


