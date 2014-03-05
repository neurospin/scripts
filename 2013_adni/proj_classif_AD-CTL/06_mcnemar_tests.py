# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:01:28 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

#import pickle
import numpy as np
#import pandas as pd
#from joblib import Parallel, delayed
import pylab as plt

#import sklearn.cross_validation
#import sklearn.linear_model
#import sklearn.linear_model.coordinate_descent
import parsimony.functions.nesterov.tv as tv

## GLOBALS ==================================================================
BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"
INPUT_PATH = os.path.join(BASE_PATH, "tv", "split")

OUTPUT_PATH = os.path.join(BASE_PATH, "tv")
INPUT_X_TRAIN_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.train.center.npy")
INPUT_X_TEST_CENTER_FILE = os.path.join(BASE_PATH, "X_CTL_AD.test.center.npy")
INPUT_Y_TRAIN_FILE = os.path.join(BASE_PATH, "y_CTL_AD.train.npy")
INPUT_Y_TEST_FILE = os.path.join(BASE_PATH, "y_CTL_AD.test.npy")

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif_AD-CTL")
sys.path.append(SRC_PATH)
import utils_proj_classif


MODE = "split"

