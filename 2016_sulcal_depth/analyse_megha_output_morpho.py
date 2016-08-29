# -*- coding: utf-8 -*-
"""
@author: yl247234
Copyrignt : CEA NeuroSpin - 2016
"""

import os, glob, re
import pheno as pu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
label_size = 22
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size 
text_size = 26


if __name__ == "__main__":    
    path = '/neurospin/brainomics/2016_sulcal_depth/STAP_output/megha/new_pheno/8th_filter/';
    for filename in glob.glob(os.path.join(path,'*stat.txt')):
        print '\n'
        print filename
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['Phenotype']
        df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
        df_sub =df.loc[np.logical_and(df['Pval'] <=1 , np.logical_not(np.isnan(df['Pval'])))]
        if not df_sub.empty:
            print df_sub

    pvaluesHan = np.zeros((3,24))
    pvalues = np.zeros((3,24))
    h2Han = np.zeros((3,24))
    h2 = np.zeros((3,24))
    count = 0
    for cap in range(54,102,2):
        filename = path + 'STAP_hull_cap'+str(cap)+'.phecovar_GenCitHan5PCA_ICV_Bv_MEGHAMEGHAstat.txt'
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['Phenotype']
        df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
        h2Han[:,count] = df['h2']
        pvaluesHan[:,count] = df['Pval']

        filename = path + 'STAP_hull_cap'+str(cap)+'.phecovar_GenCit5PCA_ICV_Bv_MEGHAMEGHAstat.txt'
        df = pd.read_csv(filename, delim_whitespace=True)
        df.index = df['Phenotype']
        df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
        h2[:,count] = df['h2']
        pvalues[:,count] = df['Pval']        
        count+=1

    subjects = np.loadtxt('/neurospin/brainomics/2016_sulcal_depth/STAP_output/megha/new_pheno/7th_filter/subjects_numbers.csv', delimiter=',')

    
    plt.figure()
    plt.plot(subjects[0,:], h2[0,:], 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
    plt.plot(subjects[0,:], h2[1,:], 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
    plt.plot(subjects[0,:], h2[2,:], 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
    plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Heritability', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.legend()
    plt.figure()
    plt.plot(subjects[1,:], h2Han[0,:], 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
    plt.plot(subjects[1,:], h2Han[1,:], 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
    plt.plot(subjects[1,:], h2Han[2,:], 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
    plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Heritability (with Handedness)', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.legend()

    plt.figure()
    plt.plot(subjects[0,:], -np.log(pvalues[0,:]), 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
    plt.plot(subjects[0,:], -np.log(pvalues[1,:]), 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
    plt.plot(subjects[0,:], -np.log(pvalues[2,:]), 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
    plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Log-Pvalues', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.legend()
    plt.figure()
    plt.plot(subjects[1,:], -np.log(pvaluesHan[0,:]), 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
    plt.plot(subjects[1,:], -np.log(pvaluesHan[1,:]), 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
    plt.plot(subjects[1,:], -np.log(pvaluesHan[2,:]), 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
    plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Log-Pvalues (with Handedness)', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.legend()


    plt.figure()
    plt.plot(range(54,102,2), subjects[0,:], 'o', markersize=7, color='blue', alpha=0.5, label='Without Handedness')
    plt.plot(range(54,102,2), subjects[1,:], 'o', markersize=7, color='green', alpha=0.5, label='With Handedness')
    plt.ylim([0,1800])
    plt.xlabel('Hull length top cap', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.ylabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
    plt.legend()


    plt.show()
