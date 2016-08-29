# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26


if __name__ == "__main__":    
    path = '/neurospin/brainomics/2016_sulcal_depth/STAP_profil/Phenotypes/ProfileExtended_session_manual_7_48_2_46_10segments/avg/allometry/megha/';
    #path = '/neurospin/brainomics/2016_sulcal_depth/STAP_profil/Phenotypes/Profile6_session_manual/allometry/megha/';
    for filename in glob.glob(os.path.join(path,'*stat.txt')):
        print '\n'
        print filename
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['Phenotype']
        df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
        df_sub =df.loc[np.logical_and(df['Pval'] <=1 , np.logical_not(np.isnan(df['Pval'])))]
        if not df_sub.empty:
            print df_sub
