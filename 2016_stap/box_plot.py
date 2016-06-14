"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import re, glob, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
label_size = 18
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
df_asym0['asymDepthMax'] = 2*(df_right0['geodesicDepthMax']-df_left0['geodesicDepthMax'])/(df_left0['geodesicDepthMax']+df_right0['geodesicDepthMax'])

#### COMPARISON MALE FEMALE ####
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=None)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'SNPSEX', u'ICV'] 
df_covar['IID'] = ['%012d' % int(i) for i in df_covar['IID']]
df_covar.index = df_covar['IID']
index_sex = df_covar.index
index_female = df_covar['IID'][df_covar['SNPSEX'] == 0]
index_male = df_covar['IID'][df_covar['SNPSEX'] == 1]

## COMPARISON RIGHT AND LEFT HANDED ###
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=None)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'Handedness', u'SNPSEX', u'ICV'] 
df_covar.index = df_covar['IID']
df_covar['IID'] = ['%012d' % int(i) for i in df_covar['IID']]
index_handed = df_covar.index
index_left = df_covar['IID'][df_covar['Handedness'] == 1]
index_right = df_covar['IID'][df_covar['Handedness'] == 0]

def customize_bp(ax, bp):
    ## Remove top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        #box.set( facecolor = '#1b9e77' )
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=2)
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)

data_to_plot = [np.asarray(df_right['geodesicDepthMax']), np.asarray(df_left['geodesicDepthMax'])]
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))
# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot
bp = ax.boxplot(data_to_plot)
customize_bp(ax, bp)
## Custom x-axis labels
ax.set_xticklabels(['Right Depth Max', 'Left Depth Max'], fontsize=text_size, fontweight = 'bold')
ax.set_ylabel('Depth Max [mm]',fontsize=text_size, fontweight = 'bold', labelpad=0)



data_to_plot2 = [np.asarray(df_asym0['asymDepthMax']), np.asarray(df_asym0['asymDepthMax'].loc[index_male]), np.asarray(df_asym0['asymDepthMax'].loc[index_female]), np.asarray(df_asym0['asymDepthMax'].loc[index_right]), np.asarray(df_asym0['asymDepthMax'].loc[index_left])]
fig = plt.figure(2, figsize=(9, 6))
# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot
bp = plt.boxplot(data_to_plot2)
customize_bp(ax, bp)
ax.set_xticklabels(['All subjects: ' + str(len(df_asym0['asymDepthMax'])), 'Male: ' + str(len(df_asym0['asymDepthMax'].loc[index_male])), 'Female: ' + str(len(df_asym0['asymDepthMax'].loc[index_female])), 'Right hand: ' + str(len(df_asym0['asymDepthMax'].loc[index_right])), 'Left hand: ' + str(len(df_asym0['asymDepthMax'].loc[index_left]))], fontsize=text_size, fontweight = 'bold')
ax.set_ylabel('AI = 2(R-L)/(R+L) ',fontsize=text_size, fontweight = 'bold', labelpad=0)

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
df_left_al = df_left.loc[df1.index]
df_left_al = df_left_al.dropna()
y = np.log(df_left_al['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['left_depthMax'] = p[0]
df_left_al['geodesicDepthMax'] = df_left_al['geodesicDepthMax']/np.power(eTIV_Bv,allometry_coeffs['left_depthMax'])
df_right_al = df_right.loc[df1.index]
df_right_al = df_right_al.dropna()
y = np.log(df_right_al['geodesicDepthMax'])
p = np.polyfit(x, y, 1)
allometry_coeffs['right_depthMax'] = p[0]
df_right_al['geodesicDepthMax'] = df_right_al['geodesicDepthMax']/np.power(eTIV_Bv,allometry_coeffs['right_depthMax'])


df_left0_al = df_left_al.loc[df_right_al.index]
df_left0_al = df_left0_al.dropna()
df_right0_al = df_right_al.loc[df_left0_al.index]
df_asym0_al = pd.DataFrame()
df_asym0_al['FID'] = df_left0_al['FID']
df_asym0_al['IID'] = df_left0_al['IID']
df_asym0_al.index = df_asym0_al['IID']
df_asym0_al['asymDepthMax'] = 2*(df_right0_al['geodesicDepthMax']-df_left0_al['geodesicDepthMax'])/(df_left0_al['geodesicDepthMax']+df_right0_al['geodesicDepthMax'])

### END ALLOMETRY NORMALISATION ####
data_to_plot3 = [np.asarray(df_asym0['asymDepthMax'].loc[index_sex]), np.asarray(df_asym0['asymDepthMax'].loc[index_male]), np.asarray(df_asym0['asymDepthMax'].loc[index_female]), np.asarray(df_asym0_al['asymDepthMax'].loc[index_sex]), np.asarray(df_asym0_al['asymDepthMax'].loc[index_male]), np.asarray(df_asym0_al['asymDepthMax'].loc[index_female])]
fig = plt.figure(3, figsize=(24, 18))
# Create an axes instance
ax = fig.add_subplot(111)
# Create the boxplot
bp = plt.boxplot(data_to_plot3)
customize_bp(ax, bp)
ax.set_xticklabels(['All: ' + str(len(df_asym0['asymDepthMax'].loc[index_sex])), 'Male: ' + str(len(df_asym0['asymDepthMax'].loc[index_male])), 'Female: ' + str(len(df_asym0['asymDepthMax'].loc[index_female])), 'All (allo)', 'Male (allo)', 'Female (allo)'], fontsize=text_size, fontweight = 'bold')
ax.set_ylabel('AI = 2(R-L)/(R+L) ',fontsize=text_size, fontweight = 'bold', labelpad=0)












plt.show()
