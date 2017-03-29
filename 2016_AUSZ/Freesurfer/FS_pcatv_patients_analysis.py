#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:12:19 2017

@author: ad247405
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
from sklearn import cluster
from sklearn import svm
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
import brainomics.image_atlas
import nibabel as nibabel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import precision_recall_fscore_support,recall_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import roc_auc_score, recall_score

INPUT_BASE_DIR = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only'
INPUT_MASK = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/mask.npy'             
DATA_Y = "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/y.npy"

y = np.load(DATA_Y) 
y = y[y!=0]

WD = "/neurospin/brainomics/2016_AUSZ/results/Freesurfer/FS_pca_tv_patients_only/model_selectionCV/all/all/struct_pca_0.1_0.8_0.1"
comp = np.load(os.path.join(WD,"components.npz"))['arr_0']
projections =  np.load(os.path.join(WD,"X_test_transform.npz"))['arr_0']

#############################################################################

#1rst component

scz = projections[y==1,0]
scz_asd = projections[y==2,0]
asd = projections[y==3,0]
data = [scz,scz_asd,asd]

import pylab as P
import numpy as np
P.figure()
bp = P.boxplot(data)
P.ylabel('Predicted')
plt.ylabel('Score on 1rst component')
P.xticks([1, 2, 3], ['SCZ', 'SCZ-ASD', 'ASD'])
for i in range(3):
    y = data[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'bo', alpha=0.6)

P.show()


#2nd component
y = np.load(DATA_Y) 
y = y[y!=0]

scz = projections[y==1,1]
scz_asd = projections[y==2,1]
asd = projections[y==3,1]
data = [scz,scz_asd,asd]

import pylab as P
import numpy as np
P.figure()
bp = P.boxplot(data)
P.ylabel('Predicted')
plt.ylabel('Score on 2nd component')
P.xticks([1, 2, 3], ['SCZ', 'SCZ-ASD', 'ASD'])
for i in range(3):
    y = data[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'bo', alpha=0.6)

P.show()


#3rd component
y = np.load(DATA_Y) 
y = y[y!=0]

scz = projections[y==1,3]
scz_asd = projections[y==2,3]
asd = projections[y==3,3]
data = [scz,scz_asd,asd]

import pylab as P
import numpy as np
P.figure()
bp = P.boxplot(data)
P.ylabel('Predicted')
plt.ylabel('Score on 3rd component')
P.xticks([1, 2, 3], ['SCZ', 'SCZ-ASD', 'ASD'])
for i in range(3):
    y = data[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'bo', alpha=0.6)

P.show()
