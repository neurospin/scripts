#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 11:20:03 2017

@author: ad247405
"""

import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt
import scipy.stats

#Before correction
###############################################################################
INPUT_DATA_X = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/X.npy"
INPUT_DATA_y = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/y.npy"

site = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/data/site.npy")
X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)


X[site==1,:].mean(axis=1)
X[site==2,:].mean(axis=1)
X[site==3,:].mean(axis=1)
X[site==4,:].mean(axis=1)
data_to_plot = [X[site==1,:].mean(axis=1),X[site==2,:].mean(axis=1),X[site==3,:].mean(axis=1),X[site==4,:].mean(axis=1)]


plt.rc('font', family='serif')
plt.figure
plt.grid()
plt.boxplot(data_to_plot)
plt.plot(np.random.normal(site, 0.04, size=site.shape[0]),X.mean(axis=1),'o',color='b',alpha = 0.4,markersize=3)
plt.xlabel("sites")
plt.ylabel("Mean Cortical Thickness across vertexes")



df = pd.DataFrame()
df["Sites"] =site
df["Mean Cortical Thickness across vertexes"] = X.mean(axis=1)
df["y"] = y
DX = {1.0: "patients", 0.0: "controls"}
df['Subjects'] = df["y"].map(DX)
import seaborn as sns, matplotlib.pyplot as plt

sns.set_style("whitegrid")
sns.set(font_scale=1.3)
ax = sns.violinplot(x="Sites", y="Mean Cortical Thickness across vertexes", hue="Subjects", data=df,\
                    split=True,linewidth = 3)
plt.tight_layout()
plt.legend(loc='lower center',ncol=2)
# statistical annotation
x1, x2 = 0, 3
y, h, col = df['Mean Cortical Thickness across vertexes'].max() + 0.05, 0.1, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h+0.01, "F = 6.0, p = 4.9e-4", ha='center', va='bottom', color=col,size = 14)


plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/Freesurfer/all_subjects/results/boxplot_FS_features.png")

scipy.stats.f_oneway(X[site==1,:].mean(axis=1),X[site==2,:].mean(axis=1),X[site==3,:].mean(axis=1),X[site==4,:].mean(axis=1))

scipy.stats.ttest_ind(X[site==1,:].mean(axis=1),X[site==2,:].mean(axis=1))
scipy.stats.ttest_ind(X[site==1,:].mean(axis=1),X[site==3,:].mean(axis=1))
scipy.stats.ttest_ind(X[site==1,:].mean(axis=1),X[site==4,:].mean(axis=1))
scipy.stats.ttest_ind(X[site==2,:].mean(axis=1),X[site==3,:].mean(axis=1))
scipy.stats.ttest_ind(X[site==2,:].mean(axis=1),X[site==4,:].mean(axis=1))
scipy.stats.ttest_ind(X[site==3,:].mean(axis=1),X[site==4,:].mean(axis=1))


