
"""
Created  11 17 2015

@author yl247235
"""

import pandas as pd
import numpy as np
import re, os, glob

import json
from genibabel import imagen_subject_ids
# Consider subjects for who we have neuroimaging and genetic data
# To fix genibabel should offer a iid function -direct request to server
login = json.load(open(os.environ['KEYPASS']))['login']
passwd = json.load(open(os.environ['KEYPASS']))['passwd']
#Set the data set of interest ("QC_Genetics", "QC_Methylation" or "QC_Expression")
data_set = "QC_Genetics"
# Since login and password are not passed, they will be requested interactily
subject_ids = imagen_subject_ids(data_of_interest=data_set, login=login,
                                 password=passwd)

path_source_BL = '/neurospin/imagen/BL/processed/nifti/'
path_source_FU2 = '/neurospin/imagen/FU2/processed/nifti/'
T1_sub_dir = 'SessionA/ADNI_MPRAGE/'
subjects_BL = [subject for subject in os.listdir(path_source_BL) if os.path.isdir(os.path.join(path_source_BL,subject))]

subjects_BL_real = []
for subject in subjects_BL:
    for filename in glob.glob(os.path.join(path_source_BL+subject+'/'+T1_sub_dir,'*nii.gz')):
        if os.path.basename(filename) == 'o'+os.path.basename(filename)[1:len(os.path.basename(filename))]:
            pass
        elif os.path.basename(filename) == 'co' +os.path.basename(filename)[2:len(os.path.basename(filename))]:
            pass
        else:
            #print filename
            """filenames2_BL.append(filename)
            subjects_catched2_BL.append(subject)"""
            subjects_BL_real.append(subject)

IID_T1_QC_gen_real = [subject for subject in subject_ids if subject in subjects_BL_real] 


directory = '/neurospin/brainomics/imagen_central/covar/'
df_sex = pd.read_csv(directory + 'plink.sexcheck', delim_whitespace=True)
#df_sex = df_sex.loc[np.logical_not(df_sex['SNPSEX']== 0)]
df3 = df_sex[['IID','SNPSEX']]
for j in range(len(df3['SNPSEX'])):
    if df3['SNPSEX'][j] == 2:
        df3['SNPSEX'][j] = 0.0
    elif df3['SNPSEX'][j] == 1:
        df3['SNPSEX'][j] = 1
    else:
        df3['SNPSEX'][j] = np.nan
for column in df3.columns:
    df3 = df3.loc[np.logical_not(np.isnan(df3[column]))]

df3['IID'] = ['%012d' % int(i) for i in df3['IID']]
df3.index = df3['IID']
df3['FID'] = df3['IID']


df_hand = pd.read_csv(directory+'request_result.csv', delimiter=';')
df11 = df_hand[['code_in_study','handedness']]
df11.columns = ['IID', 'Handness']

for j in range(len(df11['Handness'])):
    if df11['Handness'][j] == 'right':
        df11['Handness'][j] = 0.0
    elif df11['Handness'][j] == 'left':
        df11['Handness'][j] = 1.0
    else:
        df11['Handness'][j] = np.nan

temp = [np.isnan(p) for p in df11['Handness']]
df11 = df11.loc[np.logical_not(temp)]

df11['IID'] = ['%012d' % int(i) for i in df11['IID']]
df11.index = df11['IID']
df11['FID'] = df11['IID']

# All the subjects with their respective centers should be in
#/neurospin/imagen/src/scripts/psc_tools/psc2_centre.csv
# association centre-number can be found in /neurospin/imagen/RAW/PSC1/, open each centre and read its associated number
centres_number = {'4': 'Berlin','8': 'Dresden','3': 'Dublin','5': 'Hamburg','1': 'London','6': 'Mannheim','2': 'Nottingham','7': 'Paris'}
psc2_centre = np.loadtxt('/neurospin/imagen/src/scripts/psc_tools/psc2_centre.csv', delimiter=',')
all_centres_subjects = {}
for j in range(len(psc2_centre)):
    label = str(int(psc2_centre[j][0]))
    for i in range(12-len(label)):
        label = '0'+label
    all_centres_subjects[label] = centres_number[str(int(psc2_centre[j][1]))]
 

df10 = pd.DataFrame.from_dict(all_centres_subjects, orient='index')
df10.columns = ['Centres']
ldf = df10.copy()
#    print ldf.columns
colnames = ['Berlin','Dresden', 'Dublin', 'Hamburg','London', 'Mannheim', 'Nottingham', 'Paris']

colnames = ['Centres']
for c in colnames:
    #assume get_dummies add dummy cols at the end of the DF
    keep_cols = len(ldf.columns) - 1 + len(pd.unique(ldf[c])) - 1
    ldf = pd.get_dummies(ldf, columns=[c], prefix=[c])[range(keep_cols)]
#   print ldf.columns



df0= pd.DataFrame()
df0['IID'] = IID_T1_QC_gen_real
df0.index = df0['IID']

covar_path = '/neurospin/brainomics/imagen_central/covar/'
clean_covar_path = '/neurospin/brainomics/imagen_central/clean_covar/'
df2 = pd.read_csv(covar_path+'ancestry_coordinates.mds', delim_whitespace=True)
df2 = df2[['FID', 'IID', 'C1', 'C2', 'C3', 'C4', 'C5']]

df2['IID'] = ['%012d' % int(i) for i in df2['IID']]
df2.index = df2['IID']
df2['FID'] = df2['IID']
df2 = df2[['IID','FID', 'C1', 'C2', 'C3', 'C4', 'C5']]

out = 'ICV_bis.cov'
df1 = pd.read_csv(covar_path+'aseg_stats_volume_BL.csv')
df1 = df1[['Measure:volume', 'EstimatedTotalIntraCranialVol']]
df1.columns = ['IID', 'ICV']

df1['IID'] = ['%012d' % int(i) for i in df1['IID']]
df1.index = df1['IID']
df1['FID'] = df1['IID']
df1 = df1[['IID','FID','ICV']]
df1.to_csv(covar_path+out, sep= '\t', header=True, index=False)

df =df2
df[ldf.columns]=ldf[ldf.columns]
df[df3.columns]=df3[df3.columns]
df[df2.columns]=df2[df2.columns]
df[df1.columns]=df1[df1.columns]
df15 = df[[u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
       u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
       u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
           u'SNPSEX', u'ICV']]
df16 = df[[u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
       u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
       u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
           u'SNPSEX', u'ICV']]
for column in df16.columns:
    if column != 'FID' and column != 'IID':
        df16 = df16.loc[np.logical_not(np.isnan(df16[column]))]
out = 'covar_GenCit5PCA_ICV_MEGHA.cov'
df16.to_csv(clean_covar_path+out, sep= '\t', header=False, index=False)
df15[df0.columns]=df0[df0.columns]
test = []
for i in df15['IID']:
    if pd.isnull(i):
        test.append(np.nan)
    else:
        test.append(int(i))
df15['IID'] = test
for column in df15.columns:
    if column != 'FID':
        df15 = df15.loc[np.logical_not(np.isnan(df15[column]))]
df15[df11.columns]=df11[df11.columns]
temp = [np.isnan(p) for p in df15['Handness']]
df15 = df15.loc[np.logical_not(temp)]
df = df15
# The order of the columns matters for MEGHA (else 0 subjects common found
df = df[[u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
       u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
       u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham', u'Handness',
           u'SNPSEX', u'ICV']] 
df['IID'] = ['%012d' % int(i) for i in df['IID'] ]
df.index = df[u'IID']
#print df
df['FID'] = df['IID']
out = 'covar_GenCitHan5PCA_ICV_MEGHA.cov'
df.to_csv(clean_covar_path+out, sep= '\t', header=False, index=False)

df38 = df.loc[df['Handness']==0]
df38 = df38[[u'IID',u'FID', u'C1', u'C2', u'C3', u'C4', u'C5', u'Centres_Berlin',
       u'Centres_Dresden', u'Centres_Dublin', u'Centres_Hamburg',
       u'Centres_London', u'Centres_Mannheim', u'Centres_Nottingham',
           u'SNPSEX', u'ICV']] 
out = 'covar_GenCit5PCA_ICV_MEGHA_right_only.cov'
df38.to_csv(clean_covar_path+out, sep= '\t',  header=False, index=False)

# 1763 without Han (df16)
# 1565 with Han (df)
# 1383 right only (df38)
