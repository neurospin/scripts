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

columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage', 'hullJunctionsLength']
columns_f = ['FID', 'IID']+columns
from scipy.stats.stats import pearsonr
def plot_scatter(x,y, COLOR):
    """x = x/np.mean(x)
    y = y/np.mean(y)"""
    ax = plt.gca()
    ax.plot(x, y,'o', markersize=7, color=COLOR, alpha=0.5, label='Subjects')
    ax.plot(np.mean(x), np.mean(y), 'o', markersize=12, color=COLOR, alpha=1, label='Gravity center')
    p = np.polyfit(x, y, 1)
    print p
    ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color=COLOR, label='Pearson corr: '+ str(pearsonr(x,y)[0]))
    ax.legend()
    #plt.axis('equal')

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

df_right = df_right.loc[df_left.index]
df_left = df_left.loc[df_right.index]
df_right = df_right.dropna()
df_left = df_left.dropna()

df_left = df_left.loc[df2.index]
df_left = df_left.dropna()
df_right = df_right.loc[df2.index]
df_right = df_right.dropna()

x = np.log(eTIV_Bv)
y = np.log(df_left['geodesicDepthMean'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMean'] = p[0]
df_left['geodesicDepthMean'] = df_left['geodesicDepthMean']/np.power(eTIV_Bv,p[0])
y = np.log(df_left['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMax'] = p[0]
df_left['geodesicDepthMax'] = df_left['geodesicDepthMax']/np.power(eTIV_Bv,p[0])
y = np.log(df_right['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMean'] = p[0]
df_right['geodesicDepthMean'] = df_right['geodesicDepthMean']/np.power(eTIV_Bv,p[0])
y = np.log(df_right['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMax'] = p[0]
df_right['geodesicDepthMax'] = df_right['geodesicDepthMax']/np.power(eTIV_Bv,p[0])




df = pd.DataFrame()
df['asym_mean'] =(df_left['geodesicDepthMean']-df_right['geodesicDepthMean'])/(df_left['geodesicDepthMean']+df_right['geodesicDepthMean'])
df['asym_max'] =(df_left['geodesicDepthMax']-df_right['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])


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
index_female = df_covar['IID'][df_covar['SNPSEX'] == 1]
index_male = df_covar['IID'][df_covar['SNPSEX'] == 0]
index_right = df_covar['IID'][df_covar[u'Handedness'] == 0]
index_left = df_covar['IID'][df_covar[u'Handedness'] == 1]


## COMPARISON RIGHT/LEFT HANDEDNESS

fig = plt.figure()
x = df['asym_mean'].loc[index_right].dropna()
y = df['asym_max'].loc[index_right].dropna()
plot_scatter(x,y, 'blue')
x = df['asym_mean'].loc[index_left].dropna()
y = df['asym_max'].loc[index_left].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Asym depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT/LEFT HANDED', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right['geodesicDepthMean'].loc[index_right].dropna()
y = df_right['geodesicDepthMax'].loc[index_right].dropna()
plot_scatter(x,y, 'blue')
x = df_right['geodesicDepthMean'].loc[index_left].dropna()
y = df_right['geodesicDepthMax'].loc[index_left].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT/LEFT HANDED', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_left['geodesicDepthMean'].loc[index_right].dropna()
y = df_left['geodesicDepthMax'].loc[index_right].dropna()
plot_scatter(x,y, 'blue')
x = df_left['geodesicDepthMean'].loc[index_left].dropna()
y = df_left['geodesicDepthMax'].loc[index_left].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT/LEFT HANDED', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

fig = plt.figure()
x = df_right2['geodesicDepthMean'].loc[index_right].dropna()
y = df_left2['geodesicDepthMean'].loc[index_right].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMean'].loc[index_left].dropna()
y = df_left2['geodesicDepthMean'].loc[index_left].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT/LEFT HANDED', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right2['geodesicDepthMax'].loc[index_right].dropna()
y = df_left2['geodesicDepthMax'].loc[index_right].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMax'].loc[index_left].dropna()
y = df_left2['geodesicDepthMax'].loc[index_left].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT/LEFT HANDED', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")



### MALES/FEMALES


fig = plt.figure()
x = df['asym_mean'].loc[index_male].dropna()
y = df['asym_max'].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df['asym_mean'].loc[index_female].dropna()
y = df['asym_max'].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Asym depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('MALES/FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right['geodesicDepthMean'].loc[index_male].dropna()
y = df_right['geodesicDepthMax'].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right['geodesicDepthMean'].loc[index_female].dropna()
y = df_right['geodesicDepthMax'].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('MALES/FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_left['geodesicDepthMean'].loc[index_male].dropna()
y = df_left['geodesicDepthMax'].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_left['geodesicDepthMean'].loc[index_female].dropna()
y = df_left['geodesicDepthMax'].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('MALES/FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

fig = plt.figure()
x = df_right2['geodesicDepthMean'].loc[index_male].dropna()
y = df_left2['geodesicDepthMean'].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMean'].loc[index_female].dropna()
y = df_left2['geodesicDepthMean'].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('MALES/FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right2['geodesicDepthMax'].loc[index_male].dropna()
y = df_left2['geodesicDepthMax'].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMax'].loc[index_female].dropna()
y = df_left2['geodesicDepthMax'].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('MALES/FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

### RIGHT MALES/LEFT MALES


fig = plt.figure()
x = df['asym_mean'].loc[index_right].loc[index_male].dropna()
y = df['asym_max'].loc[index_right].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df['asym_mean'].loc[index_left].loc[index_male].dropna()
y = df['asym_max'].loc[index_left].loc[index_male].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Asym depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT MALES/LEFT MALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right['geodesicDepthMean'].loc[index_right].loc[index_male].dropna()
y = df_right['geodesicDepthMax'].loc[index_right].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right['geodesicDepthMean'].loc[index_left].loc[index_male].dropna()
y = df_right['geodesicDepthMax'].loc[index_left].loc[index_male].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT MALES/LEFT MALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_left['geodesicDepthMean'].loc[index_right].loc[index_male].dropna()
y = df_left['geodesicDepthMax'].loc[index_right].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_left['geodesicDepthMean'].loc[index_left].loc[index_male].dropna()
y = df_left['geodesicDepthMax'].loc[index_left].loc[index_male].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT MALES/LEFT MALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

fig = plt.figure()
x = df_right2['geodesicDepthMean'].loc[index_right].loc[index_male].dropna()
y = df_left2['geodesicDepthMean'].loc[index_right].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMean'].loc[index_left].loc[index_male].dropna()
y = df_left2['geodesicDepthMean'].loc[index_left].loc[index_male].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT MALES/LEFT MALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right2['geodesicDepthMax'].loc[index_right].loc[index_male].dropna()
y = df_left2['geodesicDepthMax'].loc[index_right].loc[index_male].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMax'].loc[index_left].loc[index_male].dropna()
y = df_left2['geodesicDepthMax'].loc[index_left].loc[index_male].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT MALES/LEFT MALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")



### RIGHT FEMALES/LEFT FEMALES


fig = plt.figure()
x = df['asym_mean'].loc[index_right].loc[index_female].dropna()
y = df['asym_max'].loc[index_right].loc[index_female].dropna()
plot_scatter(x,y, 'blue')
x = df['asym_mean'].loc[index_left].loc[index_female].dropna()
y = df['asym_max'].loc[index_left].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Asym depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Asym depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT FEMALES/LEFT FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right['geodesicDepthMean'].loc[index_right].loc[index_female].dropna()
y = df_right['geodesicDepthMax'].loc[index_right].loc[index_female].dropna()
plot_scatter(x,y, 'blue')
x = df_right['geodesicDepthMean'].loc[index_left].loc[index_female].dropna()
y = df_right['geodesicDepthMax'].loc[index_left].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT FEMALES/LEFT FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_left['geodesicDepthMean'].loc[index_right].loc[index_female].dropna()
y = df_left['geodesicDepthMax'].loc[index_right].loc[index_female].dropna()
plot_scatter(x,y, 'blue')
x = df_left['geodesicDepthMean'].loc[index_left].loc[index_female].dropna()
y = df_left['geodesicDepthMax'].loc[index_left].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT FEMALES/LEFT FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

df_right2 = df_right.loc[df_left.index]
df_left2 = df_left.loc[df_right.index]
df_right2 = df_right2.dropna()
df_left2 = df_left2.dropna()

fig = plt.figure()
x = df_right2['geodesicDepthMean'].loc[index_right].loc[index_female].dropna()
y = df_left2['geodesicDepthMean'].loc[index_right].loc[index_female].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMean'].loc[index_left].loc[index_female].dropna()
y = df_left2['geodesicDepthMean'].loc[index_left].loc[index_female].dropna()
plot_scatter(x,y, 'red')
plt.xlabel('Right depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth mean', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT FEMALES/LEFT FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

fig = plt.figure()
x = df_right2['geodesicDepthMax'].loc[index_right].loc[index_female].dropna()
y = df_left2['geodesicDepthMax'].loc[index_right].loc[index_female].dropna()
plot_scatter(x,y, 'blue')
x = df_right2['geodesicDepthMax'].loc[index_left].loc[index_female].dropna()
y = df_left2['geodesicDepthMax'].loc[index_left].loc[index_female].dropna()
plot_scatter(x,y, 'red')

plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('RIGHT FEMALES/LEFT FEMALES', fontsize =text_size+2, fontweight = 'bold', verticalalignment="bottom")

#plt.show()
