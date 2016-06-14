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

# FOR TEST SELECT HANDEDNESS ONLY
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=False)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'Handedness', u'SNPSEX', u'ICV']   
df_covar['IID']= ['%012d' % int(i) for i in df_covar['IID']]
df_covar['FID']= ['%012d' % int(i) for i in df_covar['FID']]
df_covar.index = df_covar['IID']



## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'

## OUTPUT ##
left_pheno = 'left_STAP_Hull.phe'
right_pheno = 'right_STAP_Hull.phe'
asym_pheno = 'asym_STAP_Hull.phe'

columns = ['extremity1x', 'extremity1y', 'extremity1z', 'extremity2x', 'extremity2y', 'extremity2z', 'gravityCenter_x', 'gravityCenter_y', 'gravityCenter_z', 'normal_x', 'normal_y', 'normal_z', 'direction_x', 'direction_y', 'direction_z', 'surface', 'geodesicDepthMax', 'geodesicDepthMin', 'geodesicDepthMean', 'connectedComponentsAllRels', 'connectedComponents', 'plisDePassage', 'hullJunctionsLength', 'GM_thickness' , 'pureCortical', 'fold_opening']

columns2 = ['geodesicDepthMax', 'geodesicDepthMean']


df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
import time
print "\n"
print "RIGHT"
for col in columns:
#    print df_right[col]
    print col + ":   Mean: " + str(np.mean(df_right[col])) +"   Std: "+ str(np.std(df_right[col]))



df_right = df_right.loc[df_right['hullJunctionsLength'] < 70]
df_right = df_right.loc[df_right['hullJunctionsLength'] > 40]
print "\n"
print "RIGHT"
for col in columns:
#    print df_right[col]
    print col + ":   Mean: " + str(np.mean(df_right[col])) +"   Std: "+ str(np.std(df_right[col]))
df_right = df_right[['FID','IID', 'geodesicDepthMax', 'geodesicDepthMean']]
print "\n"
for col in columns2:
    print col + ":   Mean: " + str(np.mean(df_right[col])) +"   Std: "+ str(np.std(df_right[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df_right[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('HULL FILTERED -RIGHT-'+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()

df_right.to_csv(DIRECTORY_STAP+'pheno2/'+right_pheno, sep= '\t',  header=True, index=False)

df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
print "\n"
print "LEFT"

for col in columns:
    print col + ":   Mean: " + str(np.mean(df_left[col])) +"   Std: "+ str(np.std(df_left[col]))

df_left = df_left.loc[df_left['hullJunctionsLength'] < 74]
df_left = df_left.loc[df_left['hullJunctionsLength'] > 36]
print "\n"
print "LEFT"
for col in columns:
    print col + ":   Mean: " + str(np.mean(df_left[col])) +"   Std: "+ str(np.std(df_left[col]))
df_left = df_left[['FID','IID', 'geodesicDepthMax', 'geodesicDepthMean']]
print "\n"
for col in columns2:
    print col + ":   Mean: " + str(np.mean(df_left[col])) +"   Std: "+ str(np.std(df_left[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df_left[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('HULL FILTERED -LEFT-'+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()

df_left.to_csv(DIRECTORY_STAP+'pheno2/'+left_pheno, sep= '\t',  header=True, index=False)

df = pd.DataFrame()
df['asym_mean'] =(df_left['geodesicDepthMean']-df_right['geodesicDepthMean'])/(df_left['geodesicDepthMean']+df_right['geodesicDepthMean'])
df['asym_max'] =(df_left['geodesicDepthMax']-df_right['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])
df = df.dropna()
df['IID'] = df.index
df['FID'] = df.index
df = df[['FID', 'IID', 'asym_mean', 'asym_max']]
"""df = df.loc[df_covar.index]
df = df.dropna()"""
print "\n"
for col in ['asym_mean', 'asym_max']:
    print col + ":   Mean: " + str(np.mean(df[col])) +"   Std: "+ str(np.std(df[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('HULL FILTERED -'+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()
plt.show()
df.to_csv(DIRECTORY_STAP+'pheno2/'+asym_pheno, sep= '\t',  header=True, index=False)


from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
def plot_scatter(x,y):
    plt.figure()
    plt.plot(x, y,
         'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='red', label='Pearson corr: '+ str(pearsonr(x,y)[0]))
    plt.axis('equal')
    plt.legend()

x = df['asym_mean']
y = df['asym_max']
plot_scatter(x,y)
plt.xlabel('Asym depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)

x = df_right['geodesicDepthMean']
y = df_right['geodesicDepthMax']
plot_scatter(x,y)
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)

x = df_left['geodesicDepthMean']
y = df_left['geodesicDepthMax']
plot_scatter(x,y)
plt.xlabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

x = df_right2['geodesicDepthMean']
y = df_left2['geodesicDepthMean']
plot_scatter(x,y)
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)

x = np.asarray(df_right2['geodesicDepthMax'])
y = np.asarray(df_left2['geodesicDepthMax'])
plot_scatter(x,y)
plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)

plt.show()
