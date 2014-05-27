# -*- coding: utf-8 -*-
"""
Created on Tue May 27 17:56:16 2014

@author: ed203246
"""
import os.path
IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")
INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")


import pandas

X_precomputed_train = pandas.read_csv(INPUT_ROI_TRAIN, header=None).values
y_train = pandas.read_csv(INPUT_SUBJECT_TRAIN).Label.values

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#cv = StratifiedShuffleSplit(y_train, n_iter=200, test_size=0.2)
cv = StratifiedKFold(y_train, n_folds=5)
p_lr_l2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=0.005, penalty='l2')),
])
r = cross_val_score(p_lr_l2, X_precomputed_train, y_train, cv=cv, n_jobs=-1)
print r.mean()
# 0.69999999999999996
