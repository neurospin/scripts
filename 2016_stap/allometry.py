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
DIRECTORY_STAP = '/neurospin/brainomics/2016_stap/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'


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

"""eTIV_Bv = eTIV_Bv/np.mean(eTIV_Bv)
eTIV_Freesurfer = eTIV_Freesurfer/np.mean(eTIV_Freesurfer)"""


from scipy.stats.stats import pearsonr
def plot_scatter(x,y):
    plt.figure()
    plt.plot(x, y,
         'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
    p = np.polyfit(x, y, 1)
    print p
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label='Slope:'+ str(p[0])+ ', Pearson corr: '+ str(pearsonr(x,y)[0]))
    plt.legend()

columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]
df_right['plisDePassage'] = p 
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]

df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]
df_left['plisDePassage'] = p 
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]


df = pd.DataFrame()
df['asym_mean'] =(df_left['geodesicDepthMean']-df_right['geodesicDepthMean'])/(df_left['geodesicDepthMean']+df_right['geodesicDepthMean'])
df['asym_max'] =(df_left['geodesicDepthMax']-df_right['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])

df_left2= df_left.loc[df2.index]
df_left2 = df_left2.dropna()
df_right2= df_right.loc[df2.index]
df_right2 = df_right2.dropna()

plot_scatter(np.log(eTIV_Bv), np.log(df_left2['geodesicDepthMax']))
plt.ylabel('Log Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Bv), np.log(df_right2['geodesicDepthMax']))
plt.ylabel('Log Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_left2['geodesicDepthMax']))
plt.ylabel('Log Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_right2['geodesicDepthMax']))
plt.ylabel('Log Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)


plot_scatter(np.log(eTIV_Bv), np.log(df_left2['geodesicDepthMean']))
plt.ylabel('Log Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Bv), np.log(df_right2['geodesicDepthMean']))
plt.ylabel('Log Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_left2['geodesicDepthMean']))
plt.ylabel('Log Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer),  np.log(df_right2['geodesicDepthMean']))
plt.ylabel('Log Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.close('all')
#plt.show()

left_STAP = '/neurospin/brainomics/2016_stap/brut_output/STAP_manual/STAP_left.csv'
right_STAP = '/neurospin/brainomics/2016_stap/brut_output/STAP_manual/STAP_right.csv'

columns = ['maxdepth_talairach', 'maxdepth_native']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(right_STAP)
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]

df_left = pd.read_csv(left_STAP)

df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]

df = pd.DataFrame()
df['asym_native'] =(df_left['maxdepth_native']-df_right['maxdepth_native'])/(df_left['maxdepth_native']+df_right['maxdepth_native'])
df['asym_max'] =(df_left['maxdepth_talairach']-df_right['maxdepth_talairach'])/(df_left['maxdepth_talairach']+df_right['maxdepth_talairach'])

df_left2= df_left.loc[df2.index]
df_left2 = df_left2.dropna()
df_right2= df_right.loc[df2.index]
df_right2 = df_right2.dropna()

plot_scatter(np.log(eTIV_Bv), np.log(df_left2['maxdepth_talairach']))
plt.ylabel('Log Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Bv), np.log(df_right2['maxdepth_talairach']))
plt.ylabel('Log Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_left2['maxdepth_talairach']))
plt.ylabel('Log Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_right2['maxdepth_talairach']))
plt.ylabel('Log Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)


plot_scatter(np.log(eTIV_Bv), np.log(df_left2['maxdepth_native']))
plt.ylabel('Log Left depth native', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Bv), np.log(df_right2['maxdepth_native']))
plt.ylabel('Log Right depth native', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer), np.log(df_left2['maxdepth_native']))
plt.ylabel('Log Left depth native', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(np.log(eTIV_Freesurfer),  np.log(df_right2['maxdepth_native']))
plt.ylabel('Log Right depth native', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.xlabel('Log eTIV Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)


plt.close('all')
#plt.show()
plot_scatter(df['asym_native'], df['asym_max'])

plt.xlabel('Asym depth max native', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max talairach', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.show()
