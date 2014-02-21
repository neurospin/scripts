# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:50:14 2014

@author: md238665
"""

import numpy as np
import sklearn.linear_model

#
# Learning parameters
# Used in 01_ElasticNet.py)
#

N_FOLDS = 5
FOLD_PATH_FORMAT="{fold_index}"

# Random state for CV
CV_SEED = 13031981

# Range of value for the global optimisation parameter
#  (alpha in ElasticNet)
#GLOBAL_PENALIZATION_RANGE = np.arange(0.1, 1, 0.1)
N_GLOBAL_PENALIZATION = 10

# Range of value for the l1 ratio parameter in ElasticNet
ENET_L1_RATIO_RANGE = [.1, .5, .7, .9, .95, .99, 1]

ENET_MODEL_PATH_FORMAT="{l1_ratio}-{alpha}"
