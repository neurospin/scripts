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
left_pheno = 'left_STAP.phe'
right_pheno = 'right_STAP.phe'
asym_pheno = 'asym_STAP.phe'
DIRECTORY_PHENO = DIRECTORY_STAP+'new_pheno/5th_filter/'

from scipy.stats.stats import pearsonr
def plot_scatter(x,y):
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, y,'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
    ax.plot(np.mean(x), np.mean(y), 'o', markersize=abs(np.std(x)+np.std(y)), color='green', alpha=1, label='Gravity center')
    circle1 = plt.Circle((np.mean(x),np.mean(y)), radius=1*abs(np.std(x)+np.std(y)), color='g') 
    p = np.polyfit(x, y, 1)
    print p
    ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label='Pearson corr: '+ str(pearsonr(x,y)[0]))
    ax.legend()
    plt.axis('equal')
    fig.gca().add_artist(circle1)

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

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

x = df_right2['geodesicDepthMean']
y = df_left2['geodesicDepthMean']
plot_scatter(x,y)
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
mu_x = np.mean(x)
mu_y = np.mean(y)
radius=2*abs(np.std(x)+np.std(y))
x_norm = x-mu_x
y_norm = y-mu_y
index = x_norm*x_norm+y_norm*y_norm < radius*radius # 1814 sujets/2080 0.87%

df_right2 = df_right2.loc[index]
df_left2 = df_left2.loc[index]


df2  = pd.DataFrame()
df2['asym_mean'] =(df_left2['geodesicDepthMean']-df_right2['geodesicDepthMean'])/(df_left2['geodesicDepthMean']+df_right2['geodesicDepthMean'])
df2['asym_max'] =(df_left2['geodesicDepthMax']-df_right2['geodesicDepthMax'])/(df_left2['geodesicDepthMax']+df_right2['geodesicDepthMax'])
df2['IID'] = df2.index
df2['FID'] = df2.index
df2 = df2[['FID', 'IID', 'asym_mean', 'asym_max']]

df2.to_csv(DIRECTORY_PHENO+asym_pheno, sep= '\t',  header=True, index=False)
df_right2.to_csv(DIRECTORY_PHENO+right_pheno, sep= '\t',  header=True, index=False)
df_left2.to_csv(DIRECTORY_PHENO+left_pheno, sep= '\t',  header=True, index=False)










plt.show()
