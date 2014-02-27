# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:29:56 2014

@author: md238665

Use SVM for comparison.

"""

import os
import time

import itertools

import numpy as np

import sklearn.cross_validation
import sklearn.svm

import nibabel

from joblib import Parallel, delayed

import proj_classif_config
import utils_proj_classif

BASE_PATH = "/neurospin/brainomics/2013_adni/proj_classif"

INPUT_PATH = BASE_PATH
INPUT_X_TRAIN_FILE = os.path.join(INPUT_PATH, "X_CTL_AD.train.center.npy")
INPUT_Y_TRAIN_FILE = os.path.join(INPUT_PATH, "y_CTL_AD.train.npy")
INPUT_X_TEST_FILE = os.path.join(INPUT_PATH, "X_CTL_AD.test.center.npy")
INPUT_Y_TEST_FILE = os.path.join(INPUT_PATH, "y_CTL_AD.test.npy")

INPUT_MASK_PATH = os.path.join(INPUT_PATH,
                               "SPM",
                               "template_FinalQC_CTL_AD")
INPUT_MASK = os.path.join(INPUT_MASK_PATH,
                          "mask.nii")

INPUT_PENALTIES = ['l1', 'l2']
INPUT_C = [0.1, 1, 1.0]

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

OUTPUT_PATH = os.path.join(BASE_PATH, "SVM")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#############
# Load data #
#############

X_train = np.load(INPUT_X_TRAIN_FILE)
n_train, p = X_train.shape
y_train = np.load(INPUT_Y_TRAIN_FILE)

X_test = np.load(INPUT_X_TRAIN_FILE)
n_test, _ = X_test.shape
y_test = np.load(INPUT_Y_TRAIN_FILE)

mask_im = nibabel.load(INPUT_MASK)

print p, "features"
print n_train, "training subjects"
print n_test, "testing subjects"

###############
# Launch jobs #
###############

def mapper(Xtr, ytr, Xte, yte, loss, penalty, C, dual, mask_im):
    time_curr = time.time()
    svm = sklearn.svm.LinearSVC(loss=loss,
                                penalty=penalty,
                                dual=dual,
                                C=C,
                                fit_intercept=False,
                                class_weight='auto')
    svm.fit(Xtr, ytr)
    y_pred = svm.predict(Xte)
    #y_pred = yte
    beta = svm.coef_
    #print key, "ite:%i, time:%f" % (len(mod.info["t"]), np.sum(mod.info["t"]))
    param_dir = "loss={loss}-pen={pen}-C={C}".format(loss=loss,
                                                     pen=penalty,
                                                     C=C)
    out_dir = os.path.join(OUTPUT_PATH, param_dir)
    print out_dir, "Time ellapsed:", time.time() - time_curr
    #if not os.path.exists(out_dir):
    #    os.makedirs(out_dir)
    time_curr = time.time()
    utils_proj_classif.save_model(out_dir, svm, beta, mask_im)
    np.save(os.path.join(out_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(out_dir, "y_true.npy"), yte)


Parallel(n_jobs=3, verbose=True)(
    delayed(mapper) (X_train, y_train, X_test, y_test, loss, penalty, C, dual, mask_im)
    for loss, penalty, C, dual in ALL_JOBS)
