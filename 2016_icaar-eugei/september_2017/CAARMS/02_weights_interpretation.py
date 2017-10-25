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


###############################################################################
###############################################################################

features_CAARMS_severity = np.load("/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/data/features_CAARMS_severity.npy")

weights_severity = np.load("/neurospin/brainomics/2016_icaar-eugei/september_2017/CAARMS/\
results/severity_caarms_features/svm/model_selectionCV/all/all/1e-06/beta.npz")['arr_0'].T

order = np.argsort(np.abs(weights_severity[:,0]))[::-1]


plt.rc('font', family='serif')
plt.figure
plt.grid()
fig, ax = plt.subplots()
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






