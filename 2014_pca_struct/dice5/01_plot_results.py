# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:29:39 2014

@author: md238665



"""

import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

################
# Input/Output #
################

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5/results"
INPUT_RESULTS = os.path.join(INPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

N_COMP=3
# Global penalty
GLOBAL_PENALTIES = np.array([1e-3, 1e-2, 1e-1, 1])
# Relative penalties
SRUCTPCA_L1RATIO = np.array([0.5, 1e-1, 1e-2, 1e-3, 0])
SRUCTPCA_TVRATIO = np.array([0.5, 1e-1, 1e-2, 1e-3, 0])

STRUCTPCA_PARAMS = [[l1_ratio*alpha*(1-tv_ratio),
                     (1-l1_ratio)*alpha*(1-tv_ratio),
                     alpha*tv_ratio]
                     for alpha in GLOBAL_PENALTIES
                     for tv_ratio in SRUCTPCA_TVRATIO
                     for l1_ratio in SRUCTPCA_L1RATIO]

SNRS = np.array([0.1, 0.5, 1.0])

MODEL = 'struct_pca'
MODEL_STYLE={'pca': 'o', 'sparse_pca': '--', 'struct_pca': '-'}
MODEL_COLOR={'pca': 'b', 'sparse_pca': 'g', 'struct_pca': 'r'}

##########
# Script #
##########

# Read data
total_df = pd.io.parsers.read_csv(INPUT_RESULTS)
model_df = total_df.loc[total_df['model'] == MODEL]

# Plot EVR for a given SNR and a given global penalization
width=0.8/(N_COMP)
for snr in SNRS:
    for i, global_pen in enumerate(GLOBAL_PENALTIES):
        plt.figure()
        ax = plt.gca()
        data = model_df.loc[(model_df['snr'] == snr) & (model_df['global_pen'] == global_pen)]
        grouped = data.groupby('l1_ratio')
        for name, group in grouped:
            plt.plot(SRUCTPCA_TVRATIO, group['evr_test_0'], marker='o')
        title="SNR={snr} - Global penalization={global_pen}".format(snr=snr,
                                                                    global_pen=global_pen)
        plt.title(title)
        plt.xlabel('TV ratio')
        #ax.set_xscale('log')
        plt.legend(SRUCTPCA_L1RATIO)

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
