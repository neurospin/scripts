#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 09:19:02 2017

@author: ad247405
"""

import os
import json
import numpy as np
import pandas as pd
from brainomics import array_utils
import matplotlib.pyplot as plt

INPUT_RESULTS_INTER = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/learning_curve_ratios/inter_site"

INPUT_RESULTS_INTRA = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/learning_curve_ratios/intra_site"

ratio_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
auc_inter = list()
bacc_inter = list()
bacc_se_inter = list()

auc_intra = list()
bacc_intra = list()
bacc_se_intra = list()

for i in ratio_range:
    print (i)
    results_inter = pd.read_excel(os.path.join(INPUT_RESULTS_INTER,"ratio_"+str(i),"ratio_"+str(i)+"_dcv.xlsx"),sheetname=3)
    auc_inter.append(results_inter["auc"].values)
    bacc_inter.append(results_inter["bacc"].values)
    bacc_se_inter.append(results_inter["bacc_se"].values)

    results_intra = pd.read_excel(os.path.join(INPUT_RESULTS_INTRA,"ratio_"+str(i),"ratio_"+str(i)+"_dcv.xlsx"),sheetname=3)
    auc_intra.append(results_intra["auc"].values)
    bacc_intra.append(results_intra["bacc"].values)
    bacc_se_intra.append(results_intra["bacc_se"].values)

bacc_inter = np.array(bacc_inter).ravel()
bacc_se_inter= np.array(bacc_se_inter).ravel()

bacc_intra = np.array(bacc_intra).ravel()
bacc_se_intra= np.array(bacc_se_intra).ravel()

plt.figure
plt.grid()
plt.errorbar(ratio_range,bacc_inter,bacc_se_inter, marker='o',label = "inter_site")
plt.errorbar(ratio_range,bacc_intra,bacc_se_intra, marker='^',label = "intra_site")
plt.xlabel("Ratio of training set used")
plt.ylabel("Balanced Accuracy")
plt.legend(loc="upper left")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/learning_curve_VBM.png")


plt.rc('font', family='serif')
plt.figure
plt.grid()
plt.plot(ratio_range, bacc_inter,label = "Inter-site")
plt.fill_between(ratio_range, bacc_inter-bacc_se_inter, bacc_inter+bacc_se_inter ,alpha=0.2)
plt.plot(ratio_range, bacc_intra,label = "Intra-site")
plt.fill_between(ratio_range, bacc_intra-bacc_se_intra, bacc_intra+bacc_se_intra ,alpha=0.2,color="g")
plt.xlabel("Ratio of training set used")
plt.ylabel("Balanced Accuracy")
plt.legend(loc="upper left")
plt.axis((0.1,1.0,0.50,0.80))
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/z.submission/learning_curve_VBM_filled.png")
