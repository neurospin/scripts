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

INPUT_DIR = "/neurospin/brainomics/2014_pca_struct/dice5"
INPUT_RESULTS = os.path.join(INPUT_DIR, "consolidated_results.csv")

##############
# Parameters #
##############

SNRS = np.array([0.1, 0.5, 1.0])

##########
# Script #
##########

# Read data
total_df = pd.io.parsers.read_csv(INPUT_RESULTS,
                                  index_col=[0, 1, 6, 7])

# Return indices to columns
total_df_no_ind = total_df.reset_index()
group_model_key = total_df_no_ind.groupby(['model',
                                           'Global penalization',
                                           'TV ratio'])
group_model_key_names = group_model_key.groups

COOL_GLOBAL_OPT=1.0
COOL_TV_RATIO=1e-3

PCA_MASK = total_df_no_ind['model'] == 'pca'
#GLOBAL_OPT_MASK = total_df_no_ind['Global penalization'] == COOL_GLOBAL_OPT
#TV_RATIO_MASK = (total_df_no_ind['TV ratio'] == COOL_TV_RATIO & \
#                 total_df_no_ind['model'] == 'PCA')
#SPARSE_PCA_MASK = ((total_df_no_ind['model'] == 'sparse_pca') & \
#                   (total_df_no_ind['Global penalization'] == COOL_GLOBAL_OPT))
STRUCT_PCA_MASK = ((total_df_no_ind['model'] == 'struct_pca') & \
                   (total_df_no_ind['Global penalization'] == COOL_GLOBAL_OPT) &\
                   (total_df_no_ind['TV ratio'] == COOL_TV_RATIO))


DATA=total_df_no_ind.loc[PCA_MASK | STRUCT_PCA_MASK]

SNR_GROUPS=DATA.groupby(['SNR'])

#MODEL_LIST = ['pca', 'sparse_pca', 'struct_pca']
MODEL_LIST = ['pca', 'struct_pca']
MODEL_STYLE={'pca': 'o', 'sparse_pca': '--', 'struct_pca': '-'}
MODEL_COLOR={'pca': 'b', 'sparse_pca': 'g', 'struct_pca': 'r'}
#ALPHA_COLOR={0.0: 'b', 0.1: 'b', 0.5: 'g', 1.0: 'r', 5.0: 'y'}

# Plot evr per component for each SNR
N_COMP=3
width=0.8/(N_COMP)
for snr in SNRS:
    plt.figure()
    data = SNR_GROUPS.get_group(snr)
#    plt.plot([data['evr_0'],
#              data['evr_1'],
#              data['evr_2']])
    for i, model in enumerate(MODEL_LIST):
        d = data.loc[data['model'] == model]
        plt.bar(np.arange(N_COMP)+i*width, [d['evr_0'], d['evr_1'], d['evr_2']],
                width=width, color=MODEL_COLOR[model])
    title="SNR={snr}".format(snr=snr)
    plt.title(title)
    plt.xlabel('# comp')
    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
    plt.ylabel('CPEV')
    plt.legend(MODEL_LIST, loc='upper left')
    plt.savefig(".".join(["CPEV", title, "png"]))

# Plot fscore per component for each SNR
N_COMP=3
width=0.8/(N_COMP)
for snr in SNRS:
    plt.figure()
    data = SNR_GROUPS.get_group(snr)
#    plt.plot([data['evr_0'],
#              data['evr_1'],
#              data['evr_2']])
    for i, model in enumerate(MODEL_LIST):
        d = data.loc[data['model'] == model]
        plt.bar(np.arange(N_COMP)+i*width, [d['fscore_0'],
                                            d['fscore_1'],
                                            d['fscore_2']],
                width=width, color=MODEL_COLOR[model])
    title="SNR={snr}".format(snr=snr)
    plt.title(title)
    plt.xlabel('# comp')
    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
    plt.ylabel('f_score')
    plt.legend(MODEL_LIST, loc='upper right')
    plt.savefig(".".join(["f_score", title, "png"]))

# Plot precision per component for each SNR
N_COMP=3
width=0.8/(N_COMP)
for snr in SNRS:
    plt.figure()
    data = SNR_GROUPS.get_group(snr)
#    plt.plot([data['evr_0'],
#              data['evr_1'],
#              data['evr_2']])
    for i, model in enumerate(MODEL_LIST):
        d = data.loc[data['model'] == model]
        plt.bar(np.arange(N_COMP)+i*width, [d['precision_0'],
                                            d['precision_1'],
                                            d['precision_2']],
                width=width, color=MODEL_COLOR[model])
    title="SNR={snr}".format(snr=snr)
    plt.title(title)
    plt.xlabel('# comp')
    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
    plt.ylabel('precision')
    plt.legend(MODEL_LIST, loc='upper right')
    plt.savefig(".".join(["precision", title, "png"]))

# Plot recall per component for each SNR
N_COMP=3
width=0.8/(N_COMP)
for snr in SNRS:
    plt.figure()
    data = SNR_GROUPS.get_group(snr)
#    plt.plot([data['evr_0'],
#              data['evr_1'],
#              data['evr_2']])
    for i, model in enumerate(MODEL_LIST):
        d = data.loc[data['model'] == model]
        plt.bar(np.arange(N_COMP)+i*width, [d['recall_0'],
                                            d['recall_1'],
                                            d['recall_2']],
                width=width, color=MODEL_COLOR[model])
    title="SNR={snr}".format(snr=snr)
    plt.title(title)
    plt.xlabel('# comp')
    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
    plt.ylabel('recall')
    plt.legend(MODEL_LIST, loc='upper right')
    plt.savefig(".".join(["recall", title, "png"]))

# Plot f0.25-score per component for each SNR
N_COMP=3
width=0.8/(N_COMP)
for snr in SNRS:
    plt.figure()
    data = SNR_GROUPS.get_group(snr)
#    plt.plot([data['evr_0'],
#              data['evr_1'],
#              data['evr_2']])
    for i, model in enumerate(MODEL_LIST):
        d = data.loc[data['model'] == model]
        rec = np.array([d['recall_0'], d['recall_1'], d['recall_2']])
        prec = np.array([d['precision_0'], d['precision_1'], d['precision_2']])
        beta = 2
        f2 = (1.0+beta**2)*(prec*rec)/(beta**2*prec+rec)
        plt.bar(np.arange(N_COMP)+i*width, f2,
                width=width, color=MODEL_COLOR[model])
    title="SNR={snr}".format(snr=snr)
    plt.title(title)
    plt.xlabel('# comp')
    plt.xticks(np.arange(N_COMP)+int(N_COMP/2)*width+width/2, np.arange(N_COMP))
    plt.ylabel("$F_{{{beta}}}-score$".format(beta=beta))
    plt.legend(MODEL_LIST, loc='upper right')
    plt.savefig(".".join(["fbeta_score", title, "png"]))

## Plot for each (model, key)
#plt.figure()
#for group in group_model_key_names:
#    g = group_model_key.get_group(group)
#    plt.plot(SNRS, g['evr_0'])
#plt.legend(group_model_key_names)
#
#
#COLORS=['r', 'g', 'b', 'y', 'c', 'm', 'y', 'k',
#        'aliceblue', 'azure',
#    'black', 'cornsilk',
#    'deepskyblue', 'lavenderblush',
#    'lime', 'navajowhite', 'olive',
#    'orange', 'palevioletred', 'papayawhip',
#    'royalblue', 'saddlebrown', 'tomato']
#plt.figure()
#ax = plt.gca()
#evr = None
#for snr in SNRS:
#    mask = total_df_no_ind['SNR'] == snr
#    val = total_df_no_ind['evr_0'].loc[mask].values
#    if evr is None:
#        evr = val
#    else:
#        evr = np.vstack([evr, val])
#n, p = evr.shape
#width=0.3/(p+1)
#for i in range(p):
#    plt.bar(SNRS+i*width, evr[:,i], width=width, color=COLORS[i % len(COLORS)])
#plt.xlabel('SNR')
#plt.ylabel('CEVP')
#plt.legend(group_model_key_names, loc='upper center', bbox_to_anchor=(-0.05, 1.05))
#
## Plot for each comp
#plt.figure()
#for group in group_model_key_names:
#    g = group_model_key.get_group(group)
#    g.index = SNRS
#    plt.plot([g['evr_0'][1.0],
#              g['evr_1'][1.0],
#              g['evr_2'][1.0]])
#plt.legend(group_model_key_names)
#
#MODEL_STYLE={'pca': '--', 'sparse_pca': '^', 'struct_pca': 'o'}
#ALPHA_COLOR={0.0: 'b', 0.1: 'b', 0.5: 'g', 1.0: 'r', 5.0: 'y'}
#names = []
#plt.figure()
#for group in group_model_key_names:
#    model, alpha, tv = group
#    g = group_model_key.get_group(group)
#    g.index = SNRS
#    if (model == 'struct_pca') and (tv != 1e-3):
#        continue
#    plt.plot([g['evr_0'][1.0],
#              g['evr_1'][1.0],
#              g['evr_2'][1.0]],
#              ALPHA_COLOR[alpha]+MODEL_STYLE[model])
#    names.append(group)
#plt.legend(names)

# Plot for each comp
#plt.figure()
#for group in group_model_key_names:
#    g = group_model_key.get_group(group)
#    g.index = INPUT_SNRS
#    plt.plot([g['evr_0'][0.5], g['evr_1'][0.5], g['evr_2'][0.5]])
#plt.legend(group_model_key_names)

#plt.show()