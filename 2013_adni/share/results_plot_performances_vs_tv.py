# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 00:55:29 2014

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_vbm = "/home/ed203246/Dropbox/results/2013_adni/MCIc-CTL/MCIc-CTL_cs.csv"
INPUT_fs = "/home/ed203246/Dropbox/results/2013_adni/MCIc-CTL/MCIc-CTL_csi.csv"
#INPUT = "/home/ed203246/Dropbox/results/2013_adni/MCIc-CTL-FS/MCIc-CTL-FS.csv"
INPUTS = dict(VBM=dict(filename=INPUT_vbm), FS=dict(filename=INPUT_fs))
y_col = 'recall_mean'
x_col = 'tv'
#y_col = 'auc'
a = 0.01
color_map = {0.:'#D40000', 0.1:'#F0a513',  0.5:'#2CA02C',  0.9:'#87AADE',  1.:'#214478'}
#                reds dark => brigth,      green         blues: brigth => dark

## Add points where results a shown
def add_points(data, ax):
    points = dict(# key:method, val: [tv, l1l2_ratio]
    l2=     [0,     0],
    l2tv=   [0.5,   0],
    l1=     [0,     1],
    l1tv=   [0.5,   1],
    tv=     [1,     0],
    l1l2=   [0,     0.5],
    l1l2tv= [0.3,   0.5])
    for method in points:
        tv = points[method][0]
        l1l2_ratio = points[method][1]
        d = data[(data.a == a) & (data.tv == tv) & (data.l1l2_ratio == l1l2_ratio)]
        assert d.shape[0] == 1
        #d.recall_mean
        ax.plot(tv, d[y_col], "o", color=color_map[l1l2_ratio])


# PLOT ALL RESULTS
for data_type in INPUTS:
    #print INPUT
    input_filename = INPUTS[data_type]["filename"]
    #input_filename = INPUTS[data_type]["filename"]
    outut_filename = input_filename.replace(".csv", "_"+y_col+"_x_tv_a="+str(a)+".svg")
    print outut_filename
    # Filter data
    data = pd.read_csv(input_filename)
    data.l1l2_ratio = data.l1l2_ratio.round(5)
    data = data[data.k == -1]
    data = data[data.l1l2_ratio.isin([0, 0.1, 0.5, 0.9, 1.])]
    data = data[(data.tv >= 0.1) | (data.tv == 0)]
    data = data[data.a <= 0.1]
    # for each a, l1l2_ratio, append the last point tv==1
    last = list()
    for a_ in np.unique(data.a):
        full_tv = data[(data.a == a_) & (data.tv == 1)]
        for l1l2_ratio in np.unique(data.l1l2_ratio):
            new = full_tv.copy()
            new.l1l2_ratio = l1l2_ratio
            last.append(new)
    #
    last = pd.concat(last)
    data = pd.concat([data, last])
    data.drop_duplicates(inplace=True)
    INPUTS[data_type]["data"] = data
    #
    from brainomics.plot_utilities import plot_lines_figures
    figures = plot_lines_figures(df=data,
    x_col=x_col, y_col=y_col, groupby_color_col='l1l2_ratio',
                       groupby_fig_col='a', color_map=color_map)
    # add points to figures
    for a_ in figures:
        fig = figures[a_]
        ax = fig.gca()
        add_points(data, ax)
#        for method in points:
#            tv = points[method][0]
#            l1l2_ratio = points[method][1]
#            d = data[(data.a == a) & (data.tv == tv) & (data.l1l2_ratio == l1l2_ratio)]
#            assert d.shape[0] == 1
#            d.recall_mean
#            ax.plot(tv, d[y_col], "o", color=color_map[l1l2_ratio])
    fig = figures[a_]
    ax = fig.gca()
    #plt.show()
    INPUTS[data_type]["ax"] = ax
    fig.savefig(outut_filename)

#plt.close('all')
#fig = plt.figure()
#fig.axes.append(INPUTS["VBM"]["ax"])
#fig.axes.append(INPUTS["FS"]["ax"])
#plt.show()

###############################################################################
# FINAL PLOT
plt.close('all')
plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
#fig_size =  [fig_width,fig_height]
#plt.rc('figure.figsize', [5, 5])

fig_final, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
ax1.set_xlim([-0.02, 1.02])
ax2.set_xlim([-0.02, 1.02])
ax1.set_ylim([0.645, 0.85])
ax2.set_ylim([0.645, 0.85])

#fig1.suptitle('mouse hover over figure or axes to trigger events')
#fig1.ylabel(r"Accuracy")
#fig1.set_label(["x", "y"])


vbm = INPUTS["VBM"]["data"]
print a
vbm = vbm[vbm.a == a]
vbm = vbm[(vbm.l1l2_ratio != .1) & (vbm.l1l2_ratio != .9)]

x_offset = -.01
color_groups = vbm.groupby('l1l2_ratio')
for val, group in color_groups:
    print group.shape
    group.sort(x_col, inplace=True)
    ax1.plot(group[x_col], group[y_col], label=str(val),
             color=color_map[val], linewidth=2)
    #ax1.errorbar(group[x_col]+x_offset, group[y_col], yerr=group['recall_mean_std'], color=color_map[val])#, fmt=None)
    x_offset += .01
    add_points(vbm, ax1)
    ax1.set_title("3D voxel-based GM maps")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_{\ell_1} + \lambda_{\ell_2} + \lambda_{tv})$')
    ax1.grid(True)

plt.legend()

fs = INPUTS["FS"]["data"]
fs = fs[fs.a == a]
fs = fs[(fs.l1l2_ratio != .1) & (fs.l1l2_ratio != .9)]
color_groups = fs.groupby('l1l2_ratio')
x_offset = -.01
for val, group in color_groups:
    print group.shape
    group.sort(x_col, inplace=True)
    ax2.plot(group[x_col], group[y_col], label=str(val),
             color=color_map[val], linewidth=2)
    #ax2.errorbar(group[x_col]+x_offset, group[y_col], yerr=group['recall_mean_std'], color=color_map[val])#, fmt=None)
    x_offset += .01
    add_points(fs, ax2)
    ax2.set_title("2D Cortical thickness")
    #ax2.set_ylabel("Accuracy")
    ax2.set_xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_{\ell_1} + \lambda_{\ell_2} + \lambda_{tv}$)')
    ax2.grid(True)

ax2.annotate(r'$\ell_1 \ell_2$ ratio: $\lambda_{\ell_1}/(1 - \lambda_{tv}$)', xy=(0.1, .8))
plt.legend()
plt.show()

fig_final.savefig("/home/ed203246/Dropbox/publications/2014_logistic_nestv/figures/MCIc-CTL-FS-VBM.svg")
#fig_final.savefig("/home/ed203246/Dropbox/publications/2014_logistic_nestv/figures/MCIc-CTL-FS-VBM_errbar.svg")
plt.close('all')


"""
run -i ~/git/scripts/2013_adni/share/results_plot_performances_vs_tv.py
python ~/git/scripts/2013_adni/share/results_plot_performances_vs_tv.py

"""
