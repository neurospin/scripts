"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re, argparse
import pandas as pd
sides = ['R', 'L']
feature = 'DPF'
database_parcel = 'hcp'
feature_threshold = 'DPF'

pheno_dir = '/neurospin/brainomics/2016_HCP/new_pheno_threshold_'+feature_threshold+'/pheno_pits_sym_'+feature+'_'+database_parcel+'_Freesurfer_new/'

count =0
for filename in glob.glob(os.path.join(pheno_dir,'*sideR.csv')):
    filenameL = filename[:len(filename)-5]+'L.csv'
    if os.path.isfile(filenameL):
        m = re.search(pheno_dir+feature+'_pit(.+?)sideR', filename)
        if m:
            num = m.group(1)

        df_R = pd.read_csv(filename)
        df_R.index = df_R['IID']
        df_L = pd.read_csv(filenameL)
        df_L.index = df_L['IID']
        df_L = df_L.loc[df_R.index].dropna()
        df_R = df_R.loc[df_L.index].dropna()

        df = pd.DataFrame()
        for col in df_R.columns:
            if col != 'Parcel_'+num:
                df[col] = df_R[col]
        df['Parcel_'+num] = 20*2*(df_L['Parcel_'+num]-df_R['Parcel_'+num])/(df_L['Parcel_'+num]+df_R['Parcel_'+num])
    
        if df.shape[0] > 883/2.0:
            count+=1
            print df.shape[0]
        filenameAsym = filename[:len(filename)-5]+'asym.csv'
        df.to_csv(filenameAsym,  header=True, index=False)
