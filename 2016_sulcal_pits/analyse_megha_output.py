# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import pheno as pu
import numpy as np
import pandas as pd

if __name__ == "__main__":    
    path = '/neurospin/brainomics/2016_sulcal_pits/megha/extract_v4/test1/Right/'
    for filename in glob.glob(os.path.join(path,'*stat.txt')):
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['Phenotype']
        df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
        df_sub =df.loc[np.logical_and(df['Pval'] <= 2e-1, np.logical_not(np.isnan(df['Pval'])))]
        if not df_sub.empty:
            #print '\n'
            #print filename
            print df_sub
