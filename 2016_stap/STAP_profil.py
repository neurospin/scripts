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
directory = 'Profile5_session_manual/'
number_segments = 5
right_positions = np.linspace(2., 46., num=number_segments, endpoint=True)
left_positions = np.linspace(7., 48., num=number_segments, endpoint=True)
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_profil/'
basename = 'STAP'
left_STAP = '_left.csv'
right_STAP = '_right.csv'

## OUTPUTS ##
directory_out = DIRECTORY_STAP+'Phenotypes/'+directory
if not os.path.exists(directory_out):
    os.makedirs(directory_out)



columns = ['maxdepth_talairach']
columns_f = ['FID', 'IID']+columns
df_right0 = pd.DataFrame()
df_left0 = pd.DataFrame()
df_asym0 = pd.DataFrame()
for j in range(number_segments):
    df_right = pd.read_csv(DIRECTORY_STAP+directory+basename+str(j)+right_STAP)
    
    df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
    df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
    df_right.index = df_right['IID']
    if j == 0:
        df_right0['FID'] = df_right['FID']
        df_right0['IID'] = df_right['IID']
        df_right0.index = df_right0['IID']
        #df_right0['maxdepth_talairach'] = np.zeros(len(df_right0.index))
    df_right0['maxdepth_talairach'+str(j)] = df_right['maxdepth_talairach']
    #df_right0['maxdepth_talairach'] += df_right['maxdepth_talairach']
    df_left = pd.read_csv(DIRECTORY_STAP+directory+basename+str(j)+left_STAP)
    df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
    df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
    df_left.index = df_left['IID']
    if j == 0:
        df_left0['FID'] = df_left['FID']
        df_left0['IID'] = df_left['IID']
        df_left0.index = df_left0['IID']
        #df_left0['maxdepth_talairach'] = np.zeros(len(df_left0.index))
    df_left0['maxdepth_talairach'+str(j)] = df_left['maxdepth_talairach']
    #df_left0['maxdepth_talairach'] += df_left['maxdepth_talairach']  
    

df_right0= df_right0.dropna()
df_right0.to_csv(directory_out+'right0_full.phe', sep= '\t',  header=True, index=False)
df_left0 = df_left0.dropna()
df_left0.to_csv(directory_out+'left0_full.phe', sep= '\t',  header=True, index=False)
df_right0 = df_right0.loc[df_left0.index]
df_right0= df_right0.dropna()
df_left0 = df_left0.loc[df_right0.index]
df_asym0['FID'] = df_left0['FID']
df_asym0['IID'] = df_left0['IID']
df_asym0.index = df_asym0['IID']
df_asym0['asym_mean'] = np.zeros(len(df_asym0.index))
for j in range(number_segments):
    df_asym0['maxdepth_talairach'+str(j)] = 2*(df_right0['maxdepth_talairach'+str(j)]-df_left0['maxdepth_talairach'+str(j)])/(df_left0['maxdepth_talairach'+str(j)]+df_right0['maxdepth_talairach'+str(j)])
    df_asym0['asym_mean'] += df_asym0['maxdepth_talairach'+str(j)]
df_asym0['asym_mean'] = df_asym0['asym_mean']/number_segments

df_left0.to_csv(directory_out+'left0.phe', sep= '\t',  header=True, index=False)
df_right0.to_csv(directory_out+'right0.phe', sep= '\t',  header=True, index=False)
df_asym0.to_csv(directory_out+'asym0.phe', sep= '\t',  header=True, index=False)

def plt_hist(x, COLOR, label, alpha, NORMED):
    number_bins = x.shape[0]/12
    a,b = min(x), max(x)
    n, bins, patches = plt.hist(x, number_bins, facecolor=COLOR, alpha=alpha, range=(a,b), label=label+": "+str(x.shape[0]), normed=NORMED)
"""
from scipy.stats.stats import pearsonr
def plot_scatter(x,y, COLOR, subj):
    ax = plt.gca()
    ax.plot(x, y,'-o',linewidth=10, markersize=7, color=COLOR, alpha=0.5, label=subj)

nb_curves = 0
subject_list = ['%012d' % int(i) for i in df_asym0['IID']]
colors = []
for j in range(len(subject_list)):
    colors.append(np.random.rand(3,1))

#for j in range(len(subject_list)):
for j in range(10):
     profil_depth = np.asarray(df_asym0.loc[subject_list[j]][3:])
     plot_scatter(left_positions, profil_depth, colors[j], subject_list[j])

plt.xlabel('MNI positions (mm)', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('AsymDepthMax', fontsize=text_size, fontweight = 'bold', labelpad=0)
"""
plt.figure()
plt_hist(df_asym0['asym_mean'].dropna(), 'blue','subjects', 1, False)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' ,fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Number of subjects',fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max averaged over '+ str(number_segments)+ ' segments', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
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
plt_hist(df_asym0['asym_mean'].loc[index_male].dropna(), 'blue', 'men', 1, True)            
plt_hist(df_asym0['asym_mean'].loc[index_female].dropna(), 'red','women', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max averaged over '+ str(number_segments)+ ' segments', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
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
plt_hist(df_asym0['asym_mean'].loc[index_left].dropna(), 'blue', 'left', 1, True)            
plt_hist(df_asym0['asym_mean'].loc[index_right].dropna(), 'red', 'right', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max averaged over '+ str(number_segments)+ ' segments', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

plt.show()


### ALLOMETRY NORMALIZATION ###
directory_out2 = directory_out+"allometry/"
if not os.path.exists(directory_out2):
    os.makedirs(directory_out2)
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
df2 = df2.loc[df_right0.index]
df2 = df2.dropna()
eTIV_Bv = np.asarray(df2['eTIV'])

allometry_coeffs = {}
x = np.log(eTIV_Bv)
df_right0 = df_right0.loc[df1.index]
df_right0= df_right0.dropna()
df_left0 = df_left0.loc[df1.index]
df_left0 = df_left0.dropna()
for j in range(number_segments):
    y = np.log(df_right0['maxdepth_talairach'+str(j)])
    p = np.polyfit(x, y, 1)
    allometry_coeffs['Right0_maxdepth_talairach'+str(j)] = p[0]
    df_right0['maxdepth_talairach'+str(j)] = df_right0['maxdepth_talairach'+str(j)]/np.power(eTIV_Bv,allometry_coeffs['Right0_maxdepth_talairach'+str(j)])
    y = np.log(df_left0['maxdepth_talairach'+str(j)])
    p = np.polyfit(x, y, 1)
    allometry_coeffs['Left0_maxdepth_talairach'+str(j)] = p[0]
    df_left0['maxdepth_talairach'+str(j)] = df_left0['maxdepth_talairach'+str(j)]/np.power(eTIV_Bv,allometry_coeffs['Left0_maxdepth_talairach'+str(j)])

df_right0.to_csv(directory_out2+'right0_full_allometry.phe', sep= '\t',  header=True, index=False)
df_left0.to_csv(directory_out2+'left0_full_allometry.phe', sep= '\t',  header=True, index=False)
df_right0 = df_right0.loc[df_left0.index]
df_right0= df_right0.dropna()
df_left0 = df_left0.loc[df_right0.index]
df_asym0['FID'] = df_left0['FID']
df_asym0['IID'] = df_left0['IID']
df_asym0.index = df_asym0['IID']
df_asym0['asym_mean'] = np.zeros(len(df_asym0.index))
for j in range(number_segments):
    df_asym0['maxdepth_talairach'+str(j)] = 2*(df_right0['maxdepth_talairach'+str(j)]-df_left0['maxdepth_talairach'+str(j)])/(df_left0['maxdepth_talairach'+str(j)]+df_right0['maxdepth_talairach'+str(j)])
    df_asym0['asym_mean'] += df_asym0['maxdepth_talairach'+str(j)]
df_asym0['asym_mean'] = df_asym0['asym_mean']/number_segments

df_left0.to_csv(directory_out2+'left0_allometry.phe', sep= '\t',  header=True, index=False)
df_right0.to_csv(directory_out2+'right0_allometry.phe', sep= '\t',  header=True, index=False)
df_asym0.to_csv(directory_out2+'asym0_allometry.phe', sep= '\t',  header=True, index=False)

#### COMPARISON MALE FEMALE ####
plt.figure()
plt_hist(df_asym0['asym_mean'].loc[index_male].dropna(), 'blue', 'men', 1, True)            
plt_hist(df_asym0['asym_mean'].loc[index_female].dropna(), 'red', 'women', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max averaged over '+ str(number_segments)+ ' segments (allometry normalized)', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})


## COMPARISON RIGHT AND LEFT HANDED ###
plt.figure()
plt_hist(df_asym0['asym_mean'].loc[index_left].dropna(), 'blue', 'left', 1, True)            
plt_hist(df_asym0['asym_mean'].loc[index_right].dropna(), 'red', 'right', .5, True)
plt.xlabel('Percentage difference [%] 2(R-L)/(R+L)' , fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.title('Asym of the depth max averaged over '+ str(number_segments)+ ' segments (allometry normalized)', fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
plt.legend(loc=1,prop={'size':20})

plt.show()
