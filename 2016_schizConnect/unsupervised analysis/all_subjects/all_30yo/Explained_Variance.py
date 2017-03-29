#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:20:49 2017

@author: ad247405
"""

import numpy as np
import sklearn.decomposition
from sklearn import metrics 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
import os
###############################################################################
# SCZ ONLY
############################################################################### 
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/results/pcatv_scz/5_folds_all30yo_scz"
OUTPUT_METRIC_PATH = os.path.join(BASE_PATH,"explained_variance")
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/data/X_scz.npy' 
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")


#Cumulativ explained variance ratio
cev = np.zeros((11))
#penalty start = 3
X = np.load(INPUT_DATA_X)[:,3:]
comp = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.5_0.1","components.npz"))['arr_0']
X_transform = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.5_0.1","X_test_transform.npz"))['arr_0']

#comp = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","components.npz"))['arr_0']
#X_transform = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","X_test_transform.npz"))['arr_0']

for j in range(0,11):
    X_predict = np.dot(X_transform[:,:j], comp.T[:j,:])
    cev[j] = 1 - ( (np.square(np.linalg.norm(X - X_predict, 'fro'))))  / np.square((np.linalg.norm(X, 'fro'))) 
                    

plot= plt.plot(np.arange(0,11),cev*100,'r-s',markersize=3,label = "PCA-TV")
plt.ylabel("Cumulative \n Explained Variance [\%]",fontsize=8)
plt.xlabel("Number of components",fontsize=8)

###############################################################################
# CONTROLS ONLY
############################################################################### 
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/results/pcatv_controls/5_folds_all30yo_controls"
OUTPUT_METRIC_PATH = os.path.join(BASE_PATH,"explained_variance")
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/data/X_controls.npy' 
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")


#Cumulativ explained variance ratio
cev = np.zeros((11))
#penalty start = 3
X = np.load(INPUT_DATA_X)[:,3:]
comp = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","components.npz"))['arr_0']
X_transform = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.1_0.1","X_test_transform.npz"))['arr_0']

#comp = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","components.npz"))['arr_0']
#X_transform = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","X_test_transform.npz"))['arr_0']

for j in range(0,11):
    X_predict = np.dot(X_transform[:,:j], comp.T[:j,:])
    cev[j] = 1 - ( (np.square(np.linalg.norm(X - X_predict, 'fro'))))  / np.square((np.linalg.norm(X, 'fro'))) 
                    

plot= plt.plot(np.arange(0,11),cev*100,'r-s',markersize=3,label = "PCA-TV")
plt.ylabel("Cumulative \n Explained Variance [\%]",fontsize=8)
plt.xlabel("Number of components",fontsize=8)

###############################################################################


###############################################################################
# CONTROLS + SCZ
############################################################################### 
BASE_PATH = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/results/pcatv_all/5_folds_all30yo_all"
OUTPUT_METRIC_PATH = os.path.join(BASE_PATH,"explained_variance")
INPUT_DATA_X = '/neurospin/brainomics/2016_schizConnect/analysis/all_studies/VBM/\
all_subjects_less_than_30years/data/X.npy' 
INPUT_RESULTS = os.path.join(BASE_PATH,"results","0")

#Cumulativ explained variance ratio
cev = np.zeros((11))
#penalty start = 3
X = np.load(INPUT_DATA_X)[:,3:]
comp = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.5_0.1","components.npz"))['arr_0']
X_transform = np.load(os.path.join(INPUT_RESULTS,"struct_pca_0.1_0.5_0.1","X_test_transform.npz"))['arr_0']

#comp = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","components.npz"))['arr_0']
#X_transform = np.load(os.path.join(INPUT_RESULTS,"pca_0.0_0.0_0.0","X_test_transform.npz"))['arr_0']

for j in range(0,11):
    X_predict = np.dot(X_transform[:,:j], comp.T[:j,:])
    cev[j] = 1 - ( (np.square(np.linalg.norm(X - X_predict, 'fro'))))  / np.square((np.linalg.norm(X, 'fro'))) 
                    

plot= plt.plot(np.arange(0,11),cev*100,'r-s',markersize=3,label = "PCA-TV")
plt.ylabel("Cumulative \n Explained Variance [\%]",fontsize=8)
plt.xlabel("Number of components",fontsize=8)

###############################################################################
