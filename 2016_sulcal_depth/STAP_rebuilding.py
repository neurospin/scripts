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
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_profil/'
radical = "morpho_"
left_rad = "S.T.s._left.dat"
right_rad = "S.T.s._right.dat"
number_segments = 10

df = pd.read_csv('/neurospin/brainomics/2016_sulcal_depth/STAP_profil/morpho_0S.T.s._left.dat', sep=' ')

subject_list = ['%012d' % int(i) for i in df['subject']]

from scipy.stats.stats import pearsonr
def plot_scatter(x,y, COLOR, subj):
    """x = x/np.mean(x)
    y = y/np.mean(y)"""
    ax = plt.gca()
    ax.plot(x, y,'-o',linewidth=10, markersize=7, color=COLOR, alpha=0.5, label=subj)
    #ax.plot(np.mean(x), np.mean(y), 'o', markersize=12, color=COLOR, alpha=1, label='Gravity center')
    """p = np.polyfit(x, y, 1)
    print p
    ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color=COLOR, label='Pearson corr: '+ str(pearsonr(x,y)[0]))"""
    #ax.legend()
columns = ['geodesicDepthMax', 'geodesicDepthMean']

df_rightV0 = pd.DataFrame()
df_rightV0['IID']  = subject_list
df_rightV0.index = df_rightV0['IID']
df_leftV0 = pd.DataFrame()
df_leftV0['IID'] = subject_list
df_leftV0.index = df_leftV0['IID']

for j in range(number_segments):
    df_left = pd.read_csv(DIRECTORY_STAP+radical+str(j)+left_rad, sep=' ')
    df_right = pd.read_csv(DIRECTORY_STAP+radical+str(j)+right_rad, sep=' ')
    df_left.index = ['%012d' % int(i) for i in df_left['subject']]
    df_right.index = ['%012d' % int(i) for i in df_right['subject']]
    df_leftV0[columns[0]+str(j)] = df_left[columns[0]]
    df_rightV0[columns[0]+str(j)] = df_right[columns[0]]

colors = []
for j in range(2080):
    colors.append(np.random.rand(3,1))

nb_curves = 0
for j in range(2080):
     profil_depth = np.asarray(df_leftV0.loc[subject_list[j]][1:])
     if np.count_nonzero(profil_depth) > 9:
         nb_curves += 1
         index_nonzero = np.nonzero(df_leftV0.loc[subject_list[j]][1:])
         profil_depth = profil_depth[index_nonzero]
         positions = np.linspace(12., 43., num=len(profil_depth), endpoint=True)
         plot_scatter(positions, profil_depth, colors[j], subject_list[j])

plt.xlabel('MNI positions (mm)', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('DepthMax', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.figure()

for j in range(2080):
     profil_depth = np.asarray(df_leftV0.loc[subject_list[j]][1:])
     if np.count_nonzero(profil_depth) > 9:
         index_nonzero = np.nonzero(df_leftV0.loc[subject_list[j]][1:])
         profil_depth = profil_depth[index_nonzero]
         positions = np.linspace(12., 43., num=number_segments, endpoint=True)
         positions = positions[index_nonzero]
         plot_scatter(positions, profil_depth, colors[j], subject_list[j])

plt.xlabel('MNI positions (mm)', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('DepthMax', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.figure()
nb_curves = 0
for j in range(100,200):
     profil_depth = np.asarray(df_rightV0.loc[subject_list[j]][1:])
     if np.count_nonzero(profil_depth) > 9:
         nb_curves += 1
         index_nonzero = np.nonzero(df_rightV0.loc[subject_list[j]][1:])
         profil_depth = profil_depth[index_nonzero]
         positions = np.linspace(7., 41., num=len(profil_depth), endpoint=True)
         plot_scatter(positions, profil_depth, colors[j], subject_list[j])

plt.xlabel('MNI positions (mm)', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('DepthMax', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.figure()

for j in range(100,200):
     profil_depth = np.asarray(df_rightV0.loc[subject_list[j]][1:])
     if np.count_nonzero(profil_depth) > 9:
         index_nonzero = np.nonzero(df_rightV0.loc[subject_list[j]][1:])
         profil_depth = profil_depth[index_nonzero]
         positions = np.linspace(7., 41., num=number_segments, endpoint=True)
         positions = positions[index_nonzero]
         plot_scatter(positions, profil_depth, colors[j], subject_list[j])

plt.xlabel('MNI positions (mm)', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('DepthMax', fontsize=text_size, fontweight = 'bold', labelpad=0)





plt.show()
