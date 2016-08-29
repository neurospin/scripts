# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:05:45 2016

@author: ad247405
"""

import numpy as np
import parsimony

import math
import os

import warnings
import time

from parsimony.estimators import BaseEstimator

from parsimony.algorithms.utils import AlgorithmSnapshot

import parsimony.functions as functions
import parsimony.functions.properties as properties
import parsimony.functions.nesterov as nesterov
import parsimony.utils.start_vectors as start_vectors

from parsimony.algorithms import proximal
from parsimony.algorithms.utils import Info

import parsimony.utils.consts as consts
import parsimony.utils.check_arrays as check_arrays
import parsimony.utils.maths as maths


import os, sys
import json
import time
import numpy as np
import pandas as pd
import nibabel
import argparse
import parsimony.utils.consts as consts
import pca_tv

import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.functions.nesterov.l1tv as l1tv

import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils.start_vectors as start_vectors
import brainomics.mesh_processing as mesh_utils
import sklearn
from sklearn import preprocessing
from parsimony.algorithms.utils import AlgorithmSnapshot
#import sklearn.decomposition


#
#import scipy.sparse as sparse
#
## RNG seed to get reproducible results
#np.random.seed(seed=13031981)
#
## Create data
#n = 20
#natural_shape = px, py = (100, 100)
#p = np.prod(natural_shape)
#data_shape = n, p
## Uniform data
#X = np.random.rand(n, p)
## Multiply some variables to increase variance along them
#X[:, 0] = 3*X[:, 0]
#X[:, 1] = 5*X[:, 1]
## Scale
#X = sklearn.preprocessing.scale(X, with_mean=True, with_std=False)
# # A matrices
#Atv = nesterov.tv.linear_operator_from_shape(natural_shape)
#
################################################################################
#
     

## Dataset
###############################################################################
config_filenane = "/neurospin/brainomics/2014_pca_struct/dice5_ad_validation/results_0.1_1e-6/data_100_100_0/config.json"

os.chdir(os.path.dirname(config_filenane))
config = json.load(open(config_filenane))

# Data
X = np.load(config["data"]["X"])
assert X.shape == (500,10000)
# a, l1, l2, tv penalties
global_pen = 0.01
tv_ratio = 0.5#1e-05
l1_ratio = 0.5

ltv = global_pen * tv_ratio
ll1 = l1_ratio * global_pen * (1 - tv_ratio)
ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
assert(np.allclose(ll1 + ll2 + ltv, global_pen))

#Compute A and mask
masks = []
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"
for i in range(3):
    filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
    masks.append(np.load(filename))
im_shape = config["im_shape"]
Atv = nesterov_tv.A_from_shape(im_shape)




########################################
snapshot = AlgorithmSnapshot('/neurospin/brainomics/2014_pca_struct/lambda_max/',saving_period=1).save_conesta
mod = pca_tv.PCA_L1_L2_TV(n_components=3,
                                l1=ll1, l2=ll2, ltv=ltv,
                                Atv=Atv,
                                criterion="frobenius",
                                eps=1e-4,
                                max_iter=100,
                                inner_max_iter=int(1e4),
                                output=True,callback=snapshot)  
mod.fit(X[:250,:])                                
 


