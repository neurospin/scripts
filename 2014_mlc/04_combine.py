# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:20:11 2014

@author: ed203246
"""

import os.path
import gzip
import numpy as np
import pandas as pd
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper

IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")
INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")

GM = "/neurospin/brainomics/2014_mlc/GM"
WHICH = "0.05_0.45_0.45_0.1"
penalty_start = 1
#/neurospin/brainomics/2014_mlc/GM/results/0/0.05_0.45_0.45_0.1/
y_train = pd.read_csv(INPUT_SUBJECT_TRAIN).Label.values

cv = json.load(open(os.path.join(GM, "config.json")))['resample']
#cv = StratifiedKFold(y_train, n_folds=5)
XGM = np.load(os.path.join(GM,  'GMtrain.npy'))

###############################################################################
# LOAD W check we retrive same results
alpha, l1, l2, tv = [float(p) for p in WHICH.split("_")]
l1, l2, tv = alpha * l1, alpha * l2, alpha * tv

enettv = LogisticRegressionL1L2TV(l1, l2, tv, 0, penalty_start=penalty_start,
                                   class_weight="auto")
W = list()
y_true_disk = list()
y_true_csv = list()
y_pred_disk = list()
y_pred_replay = list()

for i, (tr, te) in enumerate(cv):
    input_path = os.path.join(GM, 'results/%i/%s' % (i, WHICH))
    try:
        w = np.load(gzip.open(os.path.join(input_path, "beta.npy.gz")))
    except:
        w = np.load(os.path.join(input_path, "beta.npy"))
    enettv.beta = w
    y_pred_replay.append(enettv.predict(XGM[te,: ]).ravel())
    W.append(w.ravel())
    y_true_disk.append(np.load(os.path.join(input_path, "y_true.npy")).ravel())
    y_pred_disk.append(np.load(os.path.join(input_path, "y_pred.npy")).ravel())
    y_true_csv.append(y_train[te])

W = np.vstack(W)
y_true_disk = np.hstack(y_true_disk)
y_true_csv = np.hstack(y_true_csv)
y_pred_disk = np.hstack(y_pred_disk)
y_pred_replay = np.hstack(y_pred_replay)

# Do some QC
assert np.all(y_true_disk == y_true_csv)  # true test == those in csv
assert np.all(y_pred_replay == y_pred_disk) # pred == those stored

p, r, f, s = precision_recall_fscore_support(y_true_csv, y_pred_replay, average=None)
assert r.mean() == 0.67999999999999994
