# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:24:46 2016

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

INPUT_BASE_DIR = '/neurospin/brainomics/2016_AUSZ/results/VBM/pca'
INPUT_MASK = '/neurospin/brainomics/2016_AUSZ/results/VBM/mask.nii'             

INPUT_DIR = os.path.join(INPUT_BASE_DIR,"5_folds","results")
N_COMP = 5
EXAMPLE_FOLD = 0

babel_mask  = nibabel.load(INPUT_MASK)
mask_bool = babel_mask.get_data()
mask_bool= np.array(mask_bool !=0)
number_features = mask_bool.sum()
#

INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

params=np.array(('struct_pca', '0.1', '0.5', '0.8')) 

components = np.zeros((number_features, N_COMP))
fold=0
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

      
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]


y = np.load('/neurospin/brainomics/2016_AUSZ/results/VBM/y_patients_only.npy')
X = np.load('/neurospin/brainomics/2016_AUSZ/results/VBM/X_patients_only.npy')



#############################################################################
plt.plot(y,projections[:,0],'o')
plt.ylabel('Score on 1rst component')
plt.xlabel('group')

plt.plot(y,projections[:,1],'o')
plt.ylabel('Score on 2nd component')
plt.xlabel('group')

plt.plot(y,projections[:,2],'o')
plt.ylabel('Score on 3rd component')
plt.xlabel('group')

plt.plot(y,projections[:,3],'o')
plt.ylabel('Score on 4th component')
plt.xlabel('group')

plt.plot(y,projections[:,4],'o')
plt.ylabel('Score on 5th component')
plt.xlabel('group')


#############################################################################
import scipy
from scipy import stats

#scz vs asd
scipy.stats.ttest_ind(projections[y==1,0],projections[y==3,0])
scipy.stats.ttest_ind(projections[y==1,1],projections[y==3,1])
scipy.stats.ttest_ind(projections[y==1,2],projections[y==3,2])
scipy.stats.ttest_ind(projections[y==1,3],projections[y==3,3])
scipy.stats.ttest_ind(projections[y==1,4],projections[y==3,4])


#asd vs scz-asd
scipy.stats.ttest_ind(projections[y==1,0],projections[y==2,0])
scipy.stats.ttest_ind(projections[y==1,1],projections[y==2,1])
scipy.stats.ttest_ind(projections[y==1,2],projections[y==2,2])
scipy.stats.ttest_ind(projections[y==1,3],projections[y==2,3])
scipy.stats.ttest_ind(projections[y==1,4],projections[y==2,4])


#scz vs scz-asd
scipy.stats.ttest_ind(projections[y==3,0],projections[y==2,0])
scipy.stats.ttest_ind(projections[y==3,1],projections[y==2,1])
scipy.stats.ttest_ind(projections[y==3,2],projections[y==2,2])
scipy.stats.ttest_ind(projections[y==3,3],projections[y==2,3])
scipy.stats.ttest_ind(projections[y==3,4],projections[y==2,4])