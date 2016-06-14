"""
@author yl247234 
"""

import pandas as pd
import numpy as np
import re, glob, os

df = pd.read_csv('/neurospin/brainomics/2016_sulcal_depth/QC_stap.csv')
df2 = df[['Sujet', 'Keep', 'Problem']]
df2 = df2.dropna()
s_ids_to_remove = ['%012d' % int(i) for i in df2.loc[df2['Keep'] == 'N']['Sujet']]
s_ids = ['%012d' % int(i) for i in df2['Sujet']]
s_ids = list(set(s_ids)-set(s_ids_to_remove))

## INPUTS ##
DIRECTORY_STAP = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/'
directory = 'brut_output/'
left_STAP = 'morpho_S.T.s._left.dat'
right_STAP = 'morpho_S.T.s._right.dat'

columns = ['geodesicDepthMax', 'geodesicDepthMean', 'plisDePassage', 'hullJunctionsLength']
columns_f = ['FID', 'IID']+columns

df_right = pd.read_csv(DIRECTORY_STAP+directory+right_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_right['plisDePassage']]
df_right['plisDePassage'] = p 
df_right['FID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right['IID'] = ['%012d' % int(i) for i in df_right['subject']]
df_right.index = df_right['IID']
df_right = df_right[columns_f]

df_right = df_right.loc[s_ids]
df_right0 = df_right.loc[df_right['hullJunctionsLength']<76]

df_left = pd.read_csv(DIRECTORY_STAP+directory+left_STAP, delim_whitespace=True)
p = [2 if p >0 else 1 for p in df_left['plisDePassage']]
df_left['plisDePassage'] = p 
df_left['FID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left['IID'] = ['%012d' % int(i) for i in df_left['subject']]
df_left.index = df_left['IID']
df_left = df_left[columns_f]

df_left = df_left.loc[s_ids]
df_left0 = df_left.loc[df_left['hullJunctionsLength']<76]



columns = ['maxdepth_talairach', 'hull_junction_length_talairach']
columns_f = ['FID', 'IID']+columns
left_STAP = 'S.T.s.moy._left.csv'
right_STAP = 'S.T.s.moy._right.csv'

df_right_qc = pd.read_csv(DIRECTORY_STAP+directory+right_STAP)
df_right_qc['FID'] = ['%012d' % int(i) for i in df_right_qc['subject']]
df_right_qc['IID'] = ['%012d' % int(i) for i in df_right_qc['subject']]
df_right_qc.index = df_right_qc['IID']
df_right_qc = df_right_qc[columns_f]
df_right_qc = df_right_qc.loc[s_ids]

df_left_qc = pd.read_csv(DIRECTORY_STAP+directory+left_STAP)
df_left_qc['FID'] = ['%012d' % int(i) for i in df_left_qc['subject']]
df_left_qc['IID'] = ['%012d' % int(i) for i in df_left_qc['subject']]
df_left_qc.index = df_left_qc['IID']
df_left_qc = df_left_qc[columns_f]
df_left_qc = df_left_qc.loc[s_ids]

df_right0_qc = df_right_qc.loc[df_right_qc['hull_junction_length_talairach']<76]
df_left0_qc = df_left_qc.loc[df_left_qc['hull_junction_length_talairach']<76]



print float(df_left0.shape[0])/df_left.shape[0]
print float(df_left0_qc.shape[0])/df_left_qc.shape[0]
print float(df_right0.shape[0])/df_right.shape[0]
print float(df_right0_qc.shape[0])/df_right_qc.shape[0]
print df_left.shape[0]-df_left0.shape[0]
print float(df_left0_qc.shape[0]-df_left0.shape[0])
print float(df_left0_qc.shape[0]-df_left0.shape[0])/df_left_qc.shape[0]
print df_right.shape[0]-df_right0.shape[0]
print float(df_right0_qc.shape[0] -df_right0.shape[0])
print float(df_right0_qc.shape[0] -df_right0.shape[0])/df_right_qc.shape[0]



print "DIFF ANALYSIS"
diff_left  = df_left['geodesicDepthMax']-df_left_qc['maxdepth_talairach']
diff_left2 = diff_left.loc[diff_left != 0]
non_nul_diff_left = df_left.loc[diff_left != 0]
non_nul_diff_left_qc = df_left_qc.loc[diff_left != 0]
print non_nul_diff_left['hullJunctionsLength'].loc[non_nul_diff_left['hullJunctionsLength']<76].shape[0]
print non_nul_diff_left_qc['hull_junction_length_talairach'].loc[non_nul_diff_left_qc['hull_junction_length_talairach']<76].shape[0]
print non_nul_diff_left['hullJunctionsLength']-non_nul_diff_left_qc['hull_junction_length_talairach']

diff_right  = df_right['geodesicDepthMax']-df_right_qc['maxdepth_talairach']
non_nul_diff_right = df_right.loc[diff_right != 0]
non_nul_diff_right_qc = df_right_qc.loc[diff_right != 0]
print non_nul_diff_right['hullJunctionsLength'].loc[non_nul_diff_right['hullJunctionsLength']<76].shape[0]
print non_nul_diff_right_qc['hull_junction_length_talairach'].loc[non_nul_diff_right_qc['hull_junction_length_talairach']<76].shape[0]

print non_nul_diff_right['hullJunctionsLength']-non_nul_diff_right_qc['hull_junction_length_talairach']

"""
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

for j in range(1,9):
    subjects = []
    for s_id in s_ids:
        if all_centres_subjects[s_id] == centres_number[str(j)]:
            subjects.append(s_id)
    
    print centres_number[str(j)]
    print subjects
"""
