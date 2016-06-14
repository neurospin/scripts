

import pandas as pd
import pheno
import optparse
import re, glob, os


path = '/neurospin/brainomics/2016_sulcal_depth/pheno/all_features/length0.02/all_sulci_qc'


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
