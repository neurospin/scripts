

import pandas as pd
import pheno
import optparse
import re, glob, os


<<<<<<< HEAD
path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno0.05v2016/main_sulci_qc_all'
=======
<<<<<<< HEAD
path = '/neurospin/brainomics/2015_asym_sts/PLINK_all_pheno0.05v2016/main_sulci_qc_all'
=======
path = '/neurospin/brainomics/2016_sulcal_depth/pheno/PLINK_all_pheno0.05/all_sulci_qc'
>>>>>>> 5e917581aa7d7a6fd63a8642382b3bcf6e2143bf
>>>>>>> ac32a39e8a187ebca41d5caacc1a1e165696f891

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
