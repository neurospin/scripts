# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:29:56 2014

@author: md238665

Reload SVM prediticiton and evaluate them.

"""

import os
import itertools

import numpy as np
import pandas as pd

import sklearn.svm
from sklearn.metrics import precision_recall_fscore_support

import nibabel

import proj_classif_config
import utils_proj_classif

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"

INPUT_PATH = BASE_PATH
INPUT_Y_TEST_FILE = os.path.join(INPUT_PATH, "y_CTL_AD.test.npy")

INPUT_MASK_PATH = os.path.join(INPUT_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                          "mask.nii")

INPUT_PENALTIES = ['l1', 'l2']
INPUT_C = [0.1, 1, 10]

INPUT_SVM_PATH = os.path.join(BASE_PATH,
                              "svm")
SVM_DIR_FORMAT = "loss={loss}-pen={pen}-C={C}"

# Construct list of possible jobs (loss, penalty, C, dual)
# loss = 'l1' and penalty = 'l1' is not supported
# For loss = 'l2' and penalty = 'l1' -> dual must be False
# For loss = 'l2' and penalty = 'l2' we use False
# For loss = 'l1' and penalty = 'l2' -> dual must be True
ALL_JOBS = []
ALL_JOBS.extend(itertools.product(['l2'],
                                  INPUT_PENALTIES,
                                  INPUT_C,
                                  [False]))
ALL_JOBS.extend(itertools.product(['l1'],
                                  ['l2'],
                                  INPUT_C,
                                  [True]))

OUTPUT_PATH = INPUT_SVM_PATH
OUTPUT_FILE = os.path.join(OUTPUT_PATH,
                           "results.csv")

#############
# Load data #
#############

y_test = np.load(INPUT_Y_TEST_FILE)

mask_im = nibabel.load(INPUT_MASK)

###################
# Evaluate models #
###################

# Reload models and compute metrics
metrics = np.zeros((len(ALL_JOBS), 4))
for i, (loss, penalty, C, dual) in enumerate(ALL_JOBS):
    param_dir = SVM_DIR_FORMAT.format(loss=loss,
                                      pen=penalty,
                                      C=C)
    out_dir = os.path.join(INPUT_SVM_PATH, param_dir)
    print "Reloading", out_dir
    res = utils_proj_classif.load(out_dir)
    _, r, f, _ = precision_recall_fscore_support(res['y_true'],
                                                 res['y_pred'],
                                                 average=None)
    metrics[i] = np.hstack([r, f])

# Convert to pandas df
index = pd.MultiIndex.from_tuples([i[0:3] for i in ALL_JOBS],
                                  names=['loss', 'penalty', 'C'])
results = pd.DataFrame(metrics,
                       columns=['r0', 'r1', 'f0', 'f1'],
                       index=index)
results.to_csv(OUTPUT_FILE)
