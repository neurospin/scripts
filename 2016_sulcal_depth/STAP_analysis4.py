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
save = False

"""# FOR TEST SELECT HANDEDNESS ONLY
covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCitHan5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=False)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'Handedness', u'SNPSEX', u'ICV']   
df_covar['IID']= ['%012d' % int(i) for i in df_covar['IID']]
df_covar['FID']= ['%012d' % int(i) for i in df_covar['FID']]
df_covar.index = df_covar['IID']"""


## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'

## OUTPUT ##
left_pheno = 'left_STAP.phe'
right_pheno = 'right_STAP.phe'
asym_pheno = 'asym_STAP.phe'

#columns = ['surface', 'geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage']
columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage']



df_right = pd.read_csv(DIRECTORY_STAP+'brut_output/'+right_STAP, delim_whitespace=True)

### CHECKING IF EVERYTHING HAS BEEN FLAGGED AS VALID ##
if len(df_right['valid']) != sum(df_right['valid']):
    print "Not all subject are valid please look into it"
    df_right = df_right.loc[df_right['valid'] == 1.0]
if len(df_right['extremitiesValid']) != sum(df_right['extremitiesValid']):
    print "Not all subject have valid extremities please look into it"
    df_right = df_right.loc[df_right['extremitiesValid'] == 1.0]
if len(df_right['normalValid']) != sum(df_right['normalValid']):
    print "Not all subject have normal valid please look into it"
    df_right = df_right.loc[df_right['normalValid'] == 1.0]
## END OF CHECK ##

#df_right = df_right.loc[df_right['fold_opening'] != 10000.0] # 275sujets en moins
#df_right = df_right.loc[df_right['plisDePassage'] < 2.0] #176 sujets en moins, tous differents des 275 sujets
#df_right = df_right.loc[df_right['hullJunctionsLength'] < 100]
#df_right.loc[df_right['plisDePassage'] >= 2.0]['plisDePassage]
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]

df_right['plisDePassage'] = p 


### DISTRIBUTION HISTOGRAM ANALYSIS ###
"""
for col in ['geodesicDepthMax', 'geodesicDepthMean']:
    print df_right[col]
    print "Mean: " + str(np.mean(df_right[col]))
    print "Std: "+ str(np.std(df_right[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df_right[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('ALL SUBJECTS -RIGHT- '+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()
"""
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
columns_f = ['FID', 'IID']+columns
df_right = df_right[columns_f]
#df_right = df_right.loc[df_covar.index]
if save:
    df_right.to_csv(DIRECTORY_STAP+'pheno3/'+right_pheno, sep= '\t',  header=True, index=False)



df_left = pd.read_csv(DIRECTORY_STAP+'brut_output/'+left_STAP, delim_whitespace=True)

### CHECKING IF EVERYTHING HAS BEEN FLAGGED AS VALID ##
if len(df_left['valid']) != sum(df_left['valid']):
    print "Not all subject are valid please look into it"
    df_left = df_left.loc[df_left['valid'] == 1.0]
if len(df_left['extremitiesValid']) != sum(df_left['extremitiesValid']):
    print "Not all subject have valid extremities please look into it"
    df_left = df_left.loc[df_left['extremitiesValid'] == 1.0]
if len(df_left['normalValid']) != sum(df_left['normalValid']):
    print "Not all subject have normal valid please look into it"
    df_left = df_left.loc[df_left['normalValid'] == 1.0]
## END OF CHECK ##

#df_left = df_left.loc[df_left['fold_opening'] != 10000.0] #195sujets en moins
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]

df_left['plisDePassage'] = p  #263 sujets en moins (encore une fois tous differents des 195..)


### DISTRIBUTION HISTOGRAM ANALYSIS ###
"""import time
for col in ['geodesicDepthMax', 'geodesicDepthMean']:
    print df_left[col]
    print "Mean: " + str(np.mean(df_left[col]))
    print "Std: "+ str(np.std(df_left[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df_left[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('ALL SUBJECTS -LEFT- '+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()"""


"""mu = np.mean(df_left['geodesicDepthMax'])
sigma = np.std(df_left['geodesicDepthMax'])
df_left = df_left.loc[df_left['geodesicDepthMax'] > mu-3*sigma]
df_left = df_left.loc[df_left['geodesicDepthMax'] < mu+3*sigma]"""
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]
#df_left = df_left.loc[df_covar.index]
if save:
    df_left.to_csv(DIRECTORY_STAP+'pheno3/'+left_pheno, sep= '\t',  header=True, index=False)



# Full list columns:
"""columns = ['extremity1x', 'extremity1y', 'extremity1z', 'extremity2x', 'extremity2y', 'extremity2z', 'gravityCenter_x', 'gravityCenter_y', 'gravityCenter_z', 'normal_x', 'normal_y', 'normal_z', 'direction_x', 'direction_y', 'direction_z', 'surface', 'geodesicDepthMax', 'geodesicDepthMin', 'geodesicDepthMean', 'connectedComponentsAllRels', 'connectedComponents', 'plisDePassage', 'hullJunctionsLength', 'GM_thickness' , 'pureCortical', 'fold_opening']"""

df = pd.DataFrame()
df['asym_mean'] =(df_left['geodesicDepthMean']-df_right['geodesicDepthMean'])/(df_left['geodesicDepthMean']+df_right['geodesicDepthMean'])
df['asym_max'] =(df_left['geodesicDepthMax']-df_right['geodesicDepthMax'])/(df_left['geodesicDepthMax']+df_right['geodesicDepthMax'])
mu = np.mean(df['asym_max'])
sigma = np.std(df['asym_max'])
"""df = df.loc[df['asym_max'] > mu-2.5*sigma]
df = df.loc[df['asym_max'] < mu+2.5*sigma]"""
df = df.dropna()
df['IID'] = df.index
df['FID'] = df.index
df = df[['FID', 'IID', 'asym_mean', 'asym_max']]

df_plis = pd.DataFrame()
df_plis['asym_plis'] =  (df_left['plisDePassage']-df_right['plisDePassage'])/(df_left['plisDePassage']+df_right['plisDePassage'])
df_plis = df_plis.dropna()
df_plis['IID'] = df_plis.index
df_plis['FID'] = df_plis.index
df_plis = df_plis[['FID', 'IID', 'asym_plis']]
if save:
    df.to_csv(DIRECTORY_STAP+'pheno3/'+asym_pheno, sep= '\t',  header=True, index=False)

#df_plis.to_csv(DIRECTORY_STAP+'pheno/'+'asym_plis.phe', sep= '\t',  header=True, index=False)

"""for col in ['asym_mean', 'asym_max']:
    print df[col]
    print "Mean: " + str(np.mean(df[col]))
    print "Std: "+ str(np.std(df[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('ALL SUBJECTS -'+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()
plt.show()"""


from scipy.stats.stats import pearsonr
def plot_scatter(x,y):
    plt.figure()
    plt.plot(x, y, 'o', markersize=7, color='blue', alpha=0.5, label='Subjects')
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
plt.plot(np.mean(x), np.mean(y), 'o', markersize=7, color='green', alpha=1, label='Gravity center')
plt.xlabel('Right depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)
plt.ylabel('Left depth max', fontsize=text_size, fontweight = 'bold', labelpad=0)


plt.show()
