# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import pheno as pu
import numpy as np
import pandas as pd

pheno = ['right', 'left', 'asym']
side_to_exclude = 'zzzz' # choose a value from pheno, else write whatever doesn't match in a filename like 'zzzz'

if __name__ == "__main__":
<<<<<<< HEAD
    #path = os.path.join(os.getcwd(),'megha/all_sulci_1000000perm_fullcovar/')
    path = os.path.join(os.getcwd(), 'megha/OHBM_max_depth_more_subjects/main_sulci_qc_all_tol0.02/')
    #path =  os.path.join(os.getcwd(),'megha/OHBM_max_depth_more_subjects/all_sulci_tol0.02/')
=======
<<<<<<< HEAD
    #path = os.path.join(os.getcwd(),'megha/all_sulci_1000000perm_fullcovar/')
    path = os.path.join(os.getcwd(), 'megha/OHBM_max_depth_more_subjects/main_sulci_qc_all_tol0.02/')
    #path =  os.path.join(os.getcwd(),'megha/OHBM_max_depth_more_subjects/all_sulci_tol0.02/')
=======
    
    path = '/neurospin/brainomics/2016_sulcal_depth/megha/all_sulci_qc/tol0.05/';
>>>>>>> 5e917581aa7d7a6fd63a8642382b3bcf6e2143bf
>>>>>>> ac32a39e8a187ebca41d5caacc1a1e165696f891
    for filename in glob.glob(os.path.join(path,'*.txt')):
        print '\n'
        print filename
        if side_to_exclude not in filename:
            for j in range(len(pheno)):
                print '================ %s =================' % pheno[j]
                df = pd.read_csv(filename, delim_whitespace=True)
                df.index = df['Phenotype']
                df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
                df_sub =df.loc[np.logical_and(df['Pval'] < 1e-1, np.logical_not(np.isnan(df['Pval'])))]
                phenos = [phen for phen in df_sub.index if pheno[j] in phen]
                df_sub_sub = df_sub.loc[phenos]
                if not df_sub_sub.empty:
                    print df_sub_sub

