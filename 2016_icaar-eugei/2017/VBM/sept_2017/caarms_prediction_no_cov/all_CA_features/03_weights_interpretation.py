#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:30:41 2017

@author: ad247405
"""


import os
import numpy as np
import glob
import pandas as pd
import nibabel as nib
import brainomics.image_atlas
import shutil
import mulm
import sklearn
from  scipy import ndimage
import nibabel
import matplotlib.pyplot as plt

INPUT_CSV_ICAAR = "/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/population+scores.csv"


###############################################################################

#SVM
features_CAARMS_all = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/\
VBM/ICAAR/data/data_caarms_no_cov/features_CAARMS_all.npy")
weights_all  = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/\
VBM/ICAAR/results/CAARMS/no_cov/all_CA_features/svm/model_selectionCV/all/all/0.01/beta.npz")['arr_0'].T

order = np.argsort(np.abs(weights_all[:,0]))[::-1]

plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
# Example data
features_names = features_CAARMS_all[order[:40]]
y_pos = np.arange(len(features_names))
performance = weights_all[order[:40]]
ax.barh(y_pos, performance, color='b',alpha = 0.6,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_names,fontsize = 7)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature weights')
ax.axis('tight')
plt.tight_layout()
plt.savefig("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/results/CAARMS/no_cov/all_CA_features/svm_weights.png")
#L1
features_CAARMS_all = np.load("/neurospin/brainomics/2016_icaar-eugei/\
2017_icaar_eugei/VBM/ICAAR/data/data_with_scores/features_CAARMS_all.npy")
weights_all = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/\
results/CAARMS/CA_severity_features/l1/model_selectionCV/all/all/0.01/beta.npz")['arr_0']

order = np.argsort(np.abs(weights_all[:,0]))[::-1]

plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
# Example data
features_names = features_CAARMS_all[order[:40]]
y_pos = np.arange(len(features_names))
performance = weights_all[order[:40]]
ax.barh(y_pos, performance, color='b',alpha = 0.6,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_names,fontsize =8)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature weights')
ax.axis('tight')
plt.tight_layout()

###############################################################################
###############################################################################

features_CAARMS_severity = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/data/data_with_scores/features_CAARMS_severity.npy")
weights_severity = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/\
VBM/ICAAR/results/CAARMS/CA_severity_features/svm/model_selectionCV/all/all/1e-05/beta.npz")['arr_0'].T

order = np.argsort(np.abs(weights_severity[:,0]))[::-1]


plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
# Example data
features_names = features_CAARMS_severity[order[:30]]
y_pos = np.arange(len(features_names))
performance = weights_severity[order[:30]]
ax.barh(y_pos, performance, color='b',alpha = 0.6,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_names,fontsize =8)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature weights')
ax.axis('tight')
plt.tight_layout()



###############################################################################
###############################################################################
features_CAARMS_frequence = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/VBM/ICAAR/data/data_with_scores/features_CAARMS_frequence.npy")
weights_frequence = np.load("/neurospin/brainomics/2016_icaar-eugei/2017_icaar_eugei/\
VBM/ICAAR/results/CAARMS/CA_frequence_features/svm/model_selectionCV/all/all/0.0001/beta.npz")['arr_0'].T

order = np.argsort(np.abs(weights_frequence[:,0]))[::-1]


plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
# Example data
features_names = features_CAARMS_frequence[order[:28]]
y_pos = np.arange(len(features_names))
performance = weights_frequence[order[:28]]
ax.barh(y_pos, performance, color='b',alpha = 0.6,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(features_names,fontsize =8)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature weights')
ax.axis('tight')
plt.tight_layout()



