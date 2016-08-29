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
## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_stap/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'

columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage', 'hullJunctionsLength']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]
df_right['plisDePassage'] = p 
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]
df_right= df_right.dropna()

df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]
df_left['plisDePassage'] = p 
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]
df_left = df_left.dropna()


df_left0 = df_left.loc[df_right.index]
df_left0 = df_left0.dropna()
df_right0 = df_right.loc[df_left0.index]
df_asym0 = pd.DataFrame()
df_asym0['FID'] = df_left0['FID']
df_asym0['IID'] = df_left0['IID']
df_asym0.index = df_asym0['IID']
df_asym0['geodesicDepthMax'] = 2*(df_right0['geodesicDepthMax']-df_left0['geodesicDepthMax'])/(df_left0['geodesicDepthMax']+df_right0['geodesicDepthMax'])

def plt_hist(x, COLOR, label, alpha, NORMED):
    number_bins = x.shape[0]/15
    a,b = min(x), max(x)
    n, bins, patches = plt.hist(x, number_bins, facecolor=COLOR, alpha=alpha, range=(a,b), label=label+": "+str(x.shape[0]), normed=NORMED)



plt_hist(df_asym0['geodesicDepthMax'], 'blue', 'subjects', 1, False)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' ,fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Number of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max in STAP', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})


#### COMPARISON MALE FEMALE ####
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=None)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'SNPSEX', u'ICV'] 
df_covar['IID'] = ['%012d' % int(i) for i in df_covar['IID']]
df_covar.index = df_covar['IID']
index_female = df_covar['IID'][df_covar['SNPSEX'] == 0]
index_male = df_covar['IID'][df_covar['SNPSEX'] == 1]

plt.figure()
plt_hist(df_asym0['geodesicDepthMax'].loc[index_male], 'blue', 'men', 1, True)            
plt_hist(df_asym0['geodesicDepthMax'].loc[index_female], 'red', 'women', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max in STAP', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

## COMPARISON RIGHT AND LEFT HANDED ###
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=None)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'Handedness', u'SNPSEX', u'ICV'] 
df_covar.index = df_covar['IID']
df_covar['IID'] = ['%012d' % int(i) for i in df_covar['IID']]
index_left = df_covar['IID'][df_covar['Handedness'] == 1]
index_right = df_covar['IID'][df_covar['Handedness'] == 0]

plt.figure()
plt_hist(df_asym0['geodesicDepthMax'].loc[index_left], 'blue', 'left', 1, True)            
plt_hist(df_asym0['geodesicDepthMax'].loc[index_right], 'red', 'right', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max in STAP', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

plt.show()

### ALLOMETRY NORMALIZATION ### 
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
df2 = df2.dropna()
eTIV_Bv = np.asarray(df2['eTIV'])

allometry_coeffs = {}
x = np.log(eTIV_Bv)
df_left = df_left.loc[df1.index]
df_left = df_left.dropna()
y = np.log(df_left['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMax'] = p[0]
df_left['geodesicDepthMax'] = df_left['geodesicDepthMax']/np.power(eTIV_Bv,allometry_coeffs['left_depthMax'])
df_right = df_right.loc[df1.index]
df_right = df_right.dropna()
y = np.log(df_right['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMax'] = p[0]
df_right['geodesicDepthMax'] = df_right['geodesicDepthMax']/np.power(eTIV_Bv,allometry_coeffs['right_depthMax'])


df_left0 = df_left.loc[df_right.index]
df_left0 = df_left0.dropna()
df_right0 = df_right.loc[df_left0.index]
df_asym0 = pd.DataFrame()
df_asym0['FID'] = df_left0['FID']
df_asym0['IID'] = df_left0['IID']
df_asym0.index = df_asym0['IID']
df_asym0['geodesicDepthMax'] = 2*(df_right0['geodesicDepthMax']-df_left0['geodesicDepthMax'])/(df_left0['geodesicDepthMax']+df_right0['geodesicDepthMax'])

#### COMPARISON MALE FEMALE ####
plt.figure()
plt_hist(df_asym0['geodesicDepthMax'].loc[index_male], 'blue', 'men', 1, True)            
plt_hist(df_asym0['geodesicDepthMax'].loc[index_female], 'red', 'women', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max in STAP (allometry normalized)', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

## COMPARISON RIGHT AND LEFT HANDED ###
plt.figure()
plt_hist(df_asym0['geodesicDepthMax'].loc[index_left], 'blue', 'left', 1, True)            
plt_hist(df_asym0['geodesicDepthMax'].loc[index_right], 'red', 'right', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max in STAP (allometry normalized)', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

plt.show()
