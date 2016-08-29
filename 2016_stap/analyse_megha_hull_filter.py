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
    for j in range(7):
        if j == 5:
            HULL_MIN_VALUE = 42.0
        elif j==6:
            HULL_MIN_VALUE = 0
        else:
            HULL_MIN_VALUE = 30+j*2.5
        num_filter = 'hull_filter_min'+str(HULL_MIN_VALUE)+'/'
        path = '/neurospin/brainomics/2016_stap/Phenotypes/'+num_filter+'megha/'
        for filename in glob.glob(os.path.join(path,'*stat.txt')):
            df = pd.read_csv(filename, delim_whitespace=True)
            df.index = df['Phenotype']
            df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
            df_sub =df.loc[np.logical_and(df['Pval'] <=5e-2 , np.logical_not(np.isnan(df['Pval'])))]
            if not df_sub.empty:
                print '\n'
                print filename
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

        subjects = np.loadtxt(path+'subjects_numbers.csv', delimiter=',')


        """plt.figure()
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
        """
        plt.figure()
        plt.plot(subjects[0,:], -np.log10(pvalues[0,:]), 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
        plt.plot(subjects[0,:], -np.log10(pvalues[1,:]), 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
        plt.plot(subjects[0,:], -np.log10(pvalues[2,:]), 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
        plt.plot(range(0,2000), -np.log10(5e-2)*np.ones(2000), '-', markersize=15, color='black', alpha=1, label='Threshold 5e-2')
        plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.ylabel('Log10-Pvalues', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.title('Hull min junction length '+str(HULL_MIN_VALUE), fontsize=text_size, fontweight = 'bold')
        plt.legend()
        plt.figure()
        plt.plot(subjects[1,:], -np.log10(pvaluesHan[0,:]), 'o', markersize=7, color='red', alpha=0.5, label='Asym depth max')
        plt.plot(subjects[1,:], -np.log10(pvaluesHan[1,:]), 'o', markersize=7, color='blue', alpha=0.5, label='Left depth max')
        plt.plot(subjects[1,:], -np.log10(pvaluesHan[2,:]), 'o', markersize=7, color='green', alpha=0.5, label='Right depth max')
        plt.plot(range(0,2000), -np.log10(5e-2)*np.ones(2000), '-', markersize=15, color='black', alpha=1, label='Threshold 5e-2')
        plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.ylabel('Log10-Pvalues (with Handedness)', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.title('Hull min junction length '+str(HULL_MIN_VALUE), fontsize=text_size, fontweight = 'bold')
        plt.legend()

        """
        plt.figure()
        plt.plot(range(54,102,2), subjects[0,:], 'o', markersize=7, color='blue', alpha=0.5, label='Without Handedness')
        plt.plot(range(54,102,2), subjects[1,:], 'o', markersize=7, color='green', alpha=0.5, label='With Handedness')
        plt.ylim([0,1800])
        plt.xlabel('Hull length top cap', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.ylabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.legend()
        """

    plt.show()
