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

DIRECTORY_STAP = '/neurospin/brainomics/2016_stap/'
start_n_subj = 1100
step_n_subj = 100
n_values = 7
n_bootstrap = 100
n_phenotypes = 3
labels = ['Asym depth max', 'Left depth max', 'Right depth max']
colors = ['red', 'blue', 'green']
if __name__ == "__main__":
    for j in range(n_values):
        n_subjects = start_n_subj+j*step_n_subj
        DIRECTORY_PHENO = DIRECTORY_STAP+'Phenotypes_bootstrap_100samples/Subjects'+str(n_subjects)+'/megha_grm_update_2/'

        pvalues = np.zeros((n_phenotypes,n_bootstrap))
        h2 = np.zeros((n_phenotypes,n_bootstrap))
        for i in range(n_bootstrap):
            filename = DIRECTORY_PHENO+'STAP'+str(i)+'.phecovar_GenCit5PCA_ICV_MEGHAMEGHAstat.txt'
            df = pd.read_csv(filename, delim_whitespace=True)
            df.index = df['Phenotype']
            df = df[['h2', 'Pval', 'PermPval', 'PermFWEcPval']]
            h2[:,i] = df['h2']
            pvalues[:,i] = df['Pval']

        for k in range(n_phenotypes):
            plt.figure(k+1)
            plt.plot(np.repeat(n_subjects, n_bootstrap), -np.log10(pvalues[k,:]), 'o', markersize=7, color=colors[k], alpha=0.5, label=labels[k])
            plt.figure(k+1+n_phenotypes)
            plt.plot(np.repeat(n_subjects, n_bootstrap), h2[k,:], 'o', markersize=7, color=colors[k], alpha=0.5, label=labels[k])



    for k in range(n_phenotypes):
        plt.figure(k+1)
        plt.plot(range(start_n_subj-step_n_subj,start_n_subj+(n_values)*step_n_subj), -np.log10(5e-2)*np.ones((n_values+1)*step_n_subj), '-', markersize=15, color='black', alpha=1, label='Threshold 5e-2')
        plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.xlim( start_n_subj-step_n_subj, start_n_subj+n_values*step_n_subj)
        plt.ylabel('Log10-Pvalues', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.title('Bootstrap heritability with '+str(n_bootstrap)+' samples', fontsize=text_size, fontweight = 'bold')
        from collections import OrderedDict
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.figure(k+n_phenotypes)
        plt.xlabel('Number of subjects', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.xlim( start_n_subj-step_n_subj, start_n_subj+n_values*step_n_subj)
        plt.ylabel('Heritability', fontsize=text_size, fontweight = 'bold', labelpad=0)
        plt.title('Bootstrap heritability with '+str(n_bootstrap)+' samples', fontsize=text_size, fontweight = 'bold')
        from collections import OrderedDict
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())



    plt.show()
