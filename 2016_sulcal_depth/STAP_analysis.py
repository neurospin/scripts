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

df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
columns_f = ['FID', 'IID']+columns
df_right = df_right[columns_f]
#df_right = df_right.loc[df_covar.index]
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
#df_left = df_left.loc[df_left['hullJunctionsLength'] < 100]


### DISTRIBUTION HISTOGRAM ANALYSIS ###
import time
for col in ['geodesicDepthMax', 'geodesicDepthMean']:
    print df_left[col]
    print "Mean: " + str(np.mean(df_left[col]))
    print "Std: "+ str(np.std(df_left[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df_left[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('ALL SUBJECTS -LEFT- '+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()


"""mu = np.mean(df_left['geodesicDepthMax'])
sigma = np.std(df_left['geodesicDepthMax'])
df_left = df_left.loc[df_left['geodesicDepthMax'] > mu-3*sigma]
df_left = df_left.loc[df_left['geodesicDepthMax'] < mu+3*sigma]"""
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]
#df_left = df_left.loc[df_covar.index]
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
df.to_csv(DIRECTORY_STAP+'pheno3/'+asym_pheno, sep= '\t',  header=True, index=False)

#df_plis.to_csv(DIRECTORY_STAP+'pheno/'+'asym_plis.phe', sep= '\t',  header=True, index=False)

for col in ['asym_mean', 'asym_max']:
    print df[col]
    print "Mean: " + str(np.mean(df[col]))
    print "Std: "+ str(np.std(df[col]))
    number_bins = 100
    n, bins, patches = plt.hist(df[col], number_bins, facecolor='blue', normed=True, label=col)
    plt.xlabel('Value of the phenotype' , fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Proportion of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.title('ALL SUBJECTS -'+ col, fontsize =text_size+2, fontweight = 'bold',verticalalignment="bottom")
    plt.figure()
plt.show()



"""


covar = '/neurospin/brainomics/imagen_central/clean_covar/covar_GenCit5PCA_ICV_MEGHA.cov'
df_covar = pd.read_csv(covar, delim_whitespace=True, header=False)
df_covar.columns = [u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
                    u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
                    u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
                    u'SNPSEX', u'ICV']   
df_covar['IID']= ['%012d' % int(i) for i in df_covar['IID']]
df_covar['FID']= ['%012d' % int(i) for i in df_covar['FID']]
df_covar.index = df_covar['IID']
index_female = df_covar['IID'][df_covar['SNPSEX'] == 1]
index_male = df_covar['IID'][df_covar['SNPSEX'] == 0]


asym_depth_male = np.asarray(df.loc[index_male]['asym_max'].dropna())
asym_depth_female = np.asarray(df.loc[index_female]['asym_max'].dropna())
right_depth_male = np.asarray(df_right.loc[index_male]['geodesicDepthMax'].dropna())
right_depth_female = np.asarray(df_right.loc[index_female]['geodesicDepthMax'].dropna())
left_depth_male = np.asarray(df_left.loc[index_male]['geodesicDepthMax'].dropna())
left_depth_female = np.asarray(df_left.loc[index_female]['geodesicDepthMax'].dropna())


n_m, mu_m, std_m = len(asym_depth_male), np.mean(asym_depth_male), np.std(asym_depth_male)
n_f, mu_f, std_f = len(asym_depth_female), np.mean(asym_depth_female), np.std(asym_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Asym depth max"
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Asym t-value (saying std approximately equal): " +str(t)
print "Asym t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

n_m, mu_m, std_m = len(right_depth_male), np.mean(right_depth_male), np.std(right_depth_male)
n_f, mu_f, std_f = len(right_depth_female), np.mean(right_depth_female), np.std(right_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Right depth max"
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Right t-value (saying std approximately equal): " +str(t)
print "Right t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"


n_m, mu_m, std_m = len(left_depth_male), np.mean(left_depth_male), np.std(left_depth_male)
n_f, mu_f, std_f = len(left_depth_female), np.mean(left_depth_female), np.std(left_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
z = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)+math.pow(std_f,2))
dof = n_m+n_f-2
print "Left depth max"
print "Degree of freedom: "+ str(dof)
print "Males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Left t-value (saying std approximately equal): " +str(t)
print "Left t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

print "5% threshold for high degree of freedom is 1.65"


right_depth_male = np.asarray(df_right.loc[index_male]['geodesicDepthMax'].dropna())
right_depth_female = np.asarray(df_right.loc[index_female]['geodesicDepthMax'].dropna())
left_depth_male = np.asarray(df_left.loc[index_male]['geodesicDepthMax'].dropna())
left_depth_female = np.asarray(df_left.loc[index_female]['geodesicDepthMax'].dropna())



n_m, mu_m, std_m = len(right_depth_male), np.mean(right_depth_male), np.std(right_depth_male)
n_f, mu_f, std_f = len(left_depth_male), np.mean(left_depth_male), np.std(left_depth_male)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Males depth max"
print "Degree of freedom: "+ str(dof)
print "Right males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Left males mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Males t-value (saying std approximately equal): " +str(t)
print "Males t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

n_m, mu_m, std_m = len(right_depth_female), np.mean(right_depth_female), np.std(right_depth_female)
n_f, mu_f, std_f = len(left_depth_female), np.mean(left_depth_female), np.std(left_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
z = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)+math.pow(std_f,2))
dof = n_m+n_f-2
print "Females depth max"
print "Degree of freedom: "+ str(dof)
print "Right females mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Left females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Females t-value (saying std approximately equal): " +str(t)
print "Females t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

print "5% threshold for high degree of freedom is 1.65"


right_depth_male = np.asarray(df_right.loc[index_male]['geodesicDepthMean'].dropna())
right_depth_female = np.asarray(df_right.loc[index_female]['geodesicDepthMean'].dropna())
left_depth_male = np.asarray(df_left.loc[index_male]['geodesicDepthMean'].dropna())
left_depth_female = np.asarray(df_left.loc[index_female]['geodesicDepthMean'].dropna())



n_m, mu_m, std_m = len(right_depth_male), np.mean(right_depth_male), np.std(right_depth_male)
n_f, mu_f, std_f = len(left_depth_male), np.mean(left_depth_male), np.std(left_depth_male)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
dof = n_m+n_f-2
print "Males depth mean"
print "Degree of freedom: "+ str(dof)
print "Right males mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Left males mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Males t-value (saying std approximately equal): " +str(t)
print "Males t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

n_m, mu_m, std_m = len(right_depth_female), np.mean(right_depth_female), np.std(right_depth_female)
n_f, mu_f, std_f = len(left_depth_female), np.mean(left_depth_female), np.std(left_depth_female)
S_pooled = math.sqrt((math.pow(std_m,2)*(n_m-1)+ math.pow(std_f,2)*(n_f-1))/(n_m+n_f-2))
t = abs(mu_m-mu_f)/S_pooled*math.sqrt(n_m*n_f/(n_m+n_f))
t_unknow_std = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)/n_m+math.pow(std_f,2)/n_f)
z = abs(mu_m-mu_f)/math.sqrt(math.pow(std_m,2)+math.pow(std_f,2))
dof = n_m+n_f-2
print "Females depth mean"
print "Degree of freedom: "+ str(dof)
print "Right females mean: " + str(mu_m)+ ", std: "+ str(std_m)
print "Left females mean: " + str(mu_f)+ ", std: "+ str(std_f)
print "Females t-value (saying std approximately equal): " +str(t)
print "Females t-value (saying unknowns std): " +str(t_unknow_std)
print "\n"

print "5% threshold for high degree of freedom is 1.65"


"""
