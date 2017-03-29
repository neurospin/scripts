#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 16:44:25 2017

@author: ad247405
"""

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel as nib
from sklearn import linear_model
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn

BASE_PATH =  '/neurospin/brainomics/2016_AUSZ/results/Freesurfer'
WD = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/all_patients/regression_all_patients'

INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/X.npy'
INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/y.npy'
INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/data/mask.npy'
INPUT_CSV = '/neurospin/brainomics/2016_AUSZ/results/Freesurfer/population.csv'

n_folds = 5
    
X = np.load(INPUT_DATA_X)
y = np.load(INPUT_DATA_y)

##########################
# ASD vs CONTROLS (1 vs 0)
   ##########################
X = X[y!= 0,:]
y = y[y!= 0]
y[y==1] = 4 #temporary
y[y==2] = 1 #scz-asd
y[y==4] = 2 #asd
y[y==3] = 0#scz
assert sum(y==0) == 34 #scz
assert sum(y==1) == 22 #scz-asd
assert sum(y==2) == 36#asd


np.save(os.path.join(WD,"X.npy"),X)
np.save(os.path.join(WD,"y.npy"),y)
 
new_X = os.path.join(WD,"X.npy")
new_y = os.path.join(WD,"y.npy")


 

    
lr = linear_model.LinearRegression()

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = sklearn.cross_validation.cross_val_predict(lr,X , y, cv=5)



scz = predicted[y==0]
scz_asd = predicted[y==1]
asd = predicted[y==2]
data = [scz,scz_asd,asd]

import pylab as P
import numpy as np
P.figure()
bp = P.boxplot(data)
P.ylabel('Predicted')
P.xticks([1, 2, 3], ['SCZ', 'SCZ-ASD', 'ASD'])
for i in range(3):
    y = data[i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'bo', alpha=0.6)

P.show()
