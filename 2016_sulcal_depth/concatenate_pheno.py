

import pandas as pd
import pheno
import optparse
import re, glob, os


path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno0.05v2016/main_sulci_qc_all'

count = 0
for filename in glob.glob(os.path.join(path,'*.phe')):
    if count == 0:
        df = pd.read_csv(filename, delim_whitespace=True)
        count +=1
        df['IID'] = ['%012d' % int(i) for i in df['IID']] 
        df.index = df[u'IID']
        df['FID'] = df['IID']
    else:
        df1 = pd.read_csv(filename, delim_whitespace=True)
        count +=1
        df1['IID'] = ['%012d' % int(i) for i in df1['IID']] 
        df1.index = df1[u'IID']
        df1['FID'] = df1['IID']
        df[df1.columns]=df1[df1.columns]

out = os.path.join(path, 'concatenated_pheno.phe')
df.to_csv(out, sep='	', header=True, index=False)

no_asym_columns = [col for col in df.columns if 'asym' not in col ]
df_no_asym = df[no_asym_columns]
out = os.path.join(path, 'concatenated_pheno_without_asym.phe')
df_no_asym.to_csv(out, sep='	', header=True, index=False)
