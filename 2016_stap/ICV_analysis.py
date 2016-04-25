"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import re, glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26


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
print df1
df1['IID'] = ['%012d' % int(i) for i in df1['IID']]
df1.index = df1['IID']

df2 = df21.loc[df1.index]
df3 = df1.loc[df21.index]
df2 = df2.dropna()
df3 = df3.dropna()

eTIV_Bv = np.asarray(df2['eTIV'])
hem_closed_Bv = np.asarray(df2['hemi_closed'])
eTIV_Freesurfer = np.asarray(df3['ICV'])


eTIV_Bv = eTIV_Bv/np.mean(eTIV_Bv)
hem_closed_Bv = hem_closed_Bv/np.mean(hem_closed_Bv)
eTIV_Freesurfer = eTIV_Freesurfer/np.mean(eTIV_Freesurfer)


from scipy.stats.stats import pearsonr
def plot_scatter(x,y):
    plt.figure()
    plt.plot(x, y,
         'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label='Pearson corr: '+ str(pearsonr(x,y)[0]))
    plt.axis('equal')
    plt.legend()

plot_scatter(eTIV_Bv, hem_closed_Bv)
plt.xlabel('eTIV_Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('hem_closed_Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)

plot_scatter(eTIV_Bv, eTIV_Freesurfer)
plt.xlabel('eTIV_Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('eTIV_Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)


plot_scatter(hem_closed_Bv, eTIV_Freesurfer)
plt.xlabel('hem_closed_Bv', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('eTIV_Freesurfer', fontsize=text_size, fontweight = 'bold', labelpad=0)


plt.show()
