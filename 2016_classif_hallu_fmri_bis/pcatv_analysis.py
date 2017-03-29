# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:02:57 2016

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

INPUT_BASE_DIR = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/PCA_analysis_wto_s20'
INPUT_MASK = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results/multivariate_analysis/data/MNI152_T1_3mm_brain_mask.nii.gz'              

INPUT_DIR = os.path.join(INPUT_BASE_DIR,"5_folds","results")
N_COMP = 5
EXAMPLE_FOLD = 0

babel_mask  = nibabel.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
number_subjects = pop.shape[0]
#

INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

params=np.array(('struct_pca', '0.1', '0.1', '0.1')) 

components = np.zeros((number_features, N_COMP))
fold=0
key = '_'.join([str(param)for param in params])
print("process", key)
name=params[0]

components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

      
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

subject = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/data/subject_wto_s20.npy')
y = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/data/y_state_wto_s20.npy')
T_hallu = np.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/data/T_hallu_only_wto_s20.npy')



#############################################################################
plt.plot(subject[y==1],projections[:,0],'o')
plt.ylabel('Score on 1rst component')
plt.xlabel('Subject')

plt.plot(subject[y==1],projections[:,1],'o')
plt.ylabel('Score on 2nd component')
plt.xlabel('Subject')

plt.plot(subject[y==1],projections[:,2],'o')
plt.ylabel('Score on 3rd component')
plt.xlabel('Subject')

plt.plot(subject[y==1],projections[:,3],'o')
plt.ylabel('Score on 4th component')
plt.xlabel('Subject')

plt.plot(subject[y==1],projections[:,4],'o')
plt.ylabel('Score on 5th component')
plt.xlabel('Subject')
#############################################################################
from sklearn.cluster import KMeans
mod = KMeans(n_clusters=2)
pred = mod.fit_predict( projections[:,0].reshape(projections.shape[0],1))
plt.plot(pred,projections[:,0],'o')

#Clustering based on the 1st component score


