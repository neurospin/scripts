# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:29:39 2014

@author: md238665



"""

import os

from itertools import product

import pandas as pd
import numpy as np

import matplotlib.pylab as plt

from brainomics import plot_utilities

################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/results"
INPUT_RESULTS_FILE = os.path.join(INPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

N_COMP=3
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1])
# Relative penalties
# 0.33 ensures that there is a case with TV = L1 = L2
TVRATIO = np.array([1, 0.5, 0.33, 1e-1, 1e-2, 1e-3, 0])
L1RATIO = np.array([1, 0.5, 1e-1, 1e-2, 1e-3, 0])

PCA_PARAMS = [('pca', 0.0, 0.0, 0.0)]
SPARSE_PCA_PARAMS = list(product(['sparse_pca'],
                                 GLOBAL_PENALTIES,
                                 [0.0],
                                 [1.0]))
STRUCT_PCA_PARAMS = list(product(['struct_pca'],
                                 GLOBAL_PENALTIES,
                                 TVRATIO,
                                 L1RATIO))

SNRS = np.array([0.1, 0.5, 1.0])

MODEL = 'struct_pca'
CURVE_FILE_FORMAT = os.path.join(INPUT_DIR,
                                 'data_100_100_{snr}',
                                 '{global_pen}.svg')

##########
# Script #
##########

# Read data
df = pd.io.parsers.read_csv(INPUT_RESULTS_FILE,
                            index_col=[0, 2, 3, 4]).sort_index()
struct_pca_df = df.xs(MODEL)

# Plot some metrics for struct_pca for each SNR value
snr_groups = struct_pca_df.groupby('snr')
for snr_val, snr_group in snr_groups:
    handles = plot_utilities.mapreduce_plot(snr_group, column="recall_mean",
                                            global_pen=0,
                                            l1_ratio=2,
                                            tv_ratio=1)
    for val, handle in handles.items():
        filename = CURVE_FILE_FORMAT.format(snr=snr_val, global_pen=val)
        handle.savefig(filename)

#for snr_val, snr_group in snr_groups:
#    print "SNR:", snr_val
#    for global_pen in GLOBAL_PENALTIES:
#        print "global pen:", global_pen
#        for tv_ratio in SRUCTPCA_TVRATIO:
#            print "tv:", tv_ratio
#            data = snr_group.xs((global_pen, tv_ratio), level=[0, 1])
#            data.sort('recall_mean', inplace=True)
#            print data['recall_mean']
#            raw_input()

## Plot EVR for a given SNR and a given global penalization
#width=0.8/(N_COMP)
#for snr_val, snr_group in snr_groups:
#    for i, global_pen in enumerate(GLOBAL_PENALTIES):
#        plt.figure()
#        ax = plt.gca()
#        data = snr_group.xs(global_pen, level='global_pen')
#        grouped = data.groupby(level='l1_ratio')
#        for name, group in grouped:
#            plt.plot(TVRATIO, group['evr_test_0'], marker='o')
#        title="SNR={snr} - Global penalization={global_pen}".format(snr=snr_val,
#                                                                    global_pen=global_pen)
#        plt.title(title)
#        plt.xlabel('TV ratio')
#        #ax.set_xscale('log')
#        plt.legend(L1RATIO)

#group_key = total_df.groupby(['Global penalization',
#                              'TV ratio'])
#group_key_names = group_key.groups
#
#group_snr = total_df.groupby(['SNR'])
#
## Plot evr per component for each SNR
#width=0.8/(N_COMP)
#for snr in SNRS:
#    plt.figure()
#    data = group_snr.get_group(snr)
##    plt.plot([data['evr_0'],
##              data['evr_1'],
##              data['evr_2']])
#    for i, model in enumerate(MODEL_LIST):
#        d = data.loc[data['model'] == model]
#        plt.bar(np.arange(N_COMP)+i*width, [d['evr_shen_0'].tolist(),
#                                            d['evr_shen_1'].tolist(),
#                                            d['evr_shen_2'].tolist()],
#                width=width, color=MODEL_COLOR[model])
#    title="SNR={snr}".format(snr=snr)
#    plt.title(title)
#    plt.xlabel('# comp')
#    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
#    plt.ylabel('CPEV')
#    plt.legend(MODEL_LIST, loc='upper left')
#    plt.savefig(".".join(["CPEV", title, "png"]))
#
### Plot fscore per component for each SNR
##N_COMP=3
##width=0.8/(N_COMP)
##for snr in SNRS:
##    plt.figure()
##    data = SNR_GROUPS.get_group(snr)
###    plt.plot([data['evr_0'],
###              data['evr_1'],
###              data['evr_2']])
##    for i, model in enumerate(MODEL_LIST):
##        d = data.loc[data['model'] == model]
##        plt.bar(np.arange(N_COMP)+i*width, [d['fscore_0'],
##                                            d['fscore_1'],
##                                            d['fscore_2']],
##                width=width, color=MODEL_COLOR[model])
##    title="SNR={snr}".format(snr=snr)
##    plt.title(title)
##    plt.xlabel('# comp')
##    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
##    plt.ylabel('f_score')
##    plt.legend(MODEL_LIST, loc='upper right')
##    plt.savefig(".".join(["f_score", title, "png"]))
##
### Plot precision per component for each SNR
##N_COMP=3
##width=0.8/(N_COMP)
##for snr in SNRS:
##    plt.figure()
##    data = SNR_GROUPS.get_group(snr)
###    plt.plot([data['evr_0'],
###              data['evr_1'],
###              data['evr_2']])
##    for i, model in enumerate(MODEL_LIST):
##        d = data.loc[data['model'] == model]
##        plt.bar(np.arange(N_COMP)+i*width, [d['precision_0'],
##                                            d['precision_1'],
##                                            d['precision_2']],
##                width=width, color=MODEL_COLOR[model])
##    title="SNR={snr}".format(snr=snr)
##    plt.title(title)
##    plt.xlabel('# comp')
##    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
##    plt.ylabel('precision')
##    plt.legend(MODEL_LIST, loc='upper right')
##    plt.savefig(".".join(["precision", title, "png"]))
##
### Plot recall per component for each SNR
##N_COMP=3
##width=0.8/(N_COMP)
##for snr in SNRS:
##    plt.figure()
##    data = SNR_GROUPS.get_group(snr)
###    plt.plot([data['evr_0'],
###              data['evr_1'],
###              data['evr_2']])
##    for i, model in enumerate(MODEL_LIST):
##        d = data.loc[data['model'] == model]
##        plt.bar(np.arange(N_COMP)+i*width, [d['recall_0'],
##                                            d['recall_1'],
##                                            d['recall_2']],
##                width=width, color=MODEL_COLOR[model])
##    title="SNR={snr}".format(snr=snr)
##    plt.title(title)
##    plt.xlabel('# comp')
##    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
##    plt.ylabel('recall')
##    plt.legend(MODEL_LIST, loc='upper right')
##    plt.savefig(".".join(["recall", title, "png"]))
##
### Plot f0.25-score per component for each SNR
##N_COMP=3
##width=0.8/(N_COMP)
##for snr in SNRS:
##    plt.figure()
##    data = SNR_GROUPS.get_group(snr)
###    plt.plot([data['evr_0'],
###              data['evr_1'],
###              data['evr_2']])
##    for i, model in enumerate(MODEL_LIST):
##        d = data.loc[data['model'] == model]
##        rec = np.array([d['recall_0'], d['recall_1'], d['recall_2']])
##        prec = np.array([d['precision_0'], d['precision_1'], d['precision_2']])
##        beta = 2
##        f2 = (1.0+beta**2)*(prec*rec)/(beta**2*prec+rec)
##        plt.bar(np.arange(N_COMP)+i*width, f2,
##                width=width, color=MODEL_COLOR[model])
##    title="SNR={snr}".format(snr=snr)
##    plt.title(title)
##    plt.xlabel('# comp')
##    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
##    plt.ylabel("$F_{{{beta}}}-score$".format(beta=beta))
##    plt.legend(MODEL_LIST, loc='upper right')
##    plt.savefig(".".join(["fbeta_score", title, "png"]))
