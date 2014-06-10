# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:20:11 2014

@author: ed203246
run scripts/2014_mlc/04_combine.py
"""

import os.path, sys
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
import nibabel

IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")
INPUT_ROI_TEST = os.path.join(IMAGES_PATH, "BinaryTest_data.csv")

INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")
INPUT_SUBJECT_TEST = os.path.join(IMAGES_PATH, "BinaryTest_sbj_list.csv")

SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2014_mlc")
sys.path.append(SRC_PATH)
import utils


WHICH = "0.05_0.45_0.45_0.1"
GM = "/neurospin/brainomics/2014_mlc/GM"

penalty_start = 1
#/neurospin/brainomics/2014_mlc/GM/results/0/0.05_0.45_0.45_0.1/
pop_train = pd.read_csv(INPUT_SUBJECT_TRAIN)
pop_test = pd.read_csv(INPUT_SUBJECT_TEST)

#cv = json.load(open(os.path.join(GM, "config.json")))['resample']
#cv = StratifiedKFold(y_train, n_folds=5)
Xgm_train = np.load(os.path.join(GM,  'GMtrain.npy'))
Xgm_test = np.load(os.path.join(GM,  'GMtest.npy'))
y_train = np.load(os.path.join(GM,  'ytrain.npy'))
assert np.all(pop_train.Label.values  == y_train.ravel())

Xroi_train = pd.read_csv(INPUT_ROI_TRAIN, header=None).values
Xroi_test = pd.read_csv(INPUT_ROI_TEST, header=None).values
assert Xroi_train.shape == (150, 184)
assert Xroi_test.shape == (100, 184)
assert Xgm_train.shape == (150, 379778)
assert Xgm_test.shape == (100, 379778)


# enettv for GM
arg = [float(p) for p in WHICH.split("_")]
if len(arg) == 4:
    alpha, l1, l2, tv = arg
else:
    alpha, l1, l2, tv, k = arg



l1, l2, tv = alpha * l1, alpha * l2, alpha * tv
mask_im = nibabel.load(os.path.join(GM,  "mask.nii"))
A, _ = tv_helper.A_from_mask(mask_im.get_data())
enettv = LogisticRegressionL1L2TV(l1, l2, tv, A, penalty_start=penalty_start,
                                   class_weight="auto")

# lr l2 for roi
p_lr_l2 = Pipeline([
    ('scaler', StandardScaler()),
   # ('classifier', LogisticRegression(C=0.005, penalty='l2')),
    ('classifier', LogisticRegression(C=0.0022, penalty='l2')),
])


print "==============================================="
print "== Refit on all data to predict test samples =="
print "==============================================="
# Train, Input
# y_train, Xgm_train
# Test Input Xgm_test
assert np.all(pop_train.Label.values  == y_train.ravel())
assert Xgm_test.shape == (100, Xgm_train.shape[1]) 
assert np.all(pop_train.Label.values == y_train.ravel())

OUTPUT = os.path.join(os.path.dirname(GM), 'combine', "gm_all_"+WHICH)
if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)
#Xtr = np.load(os.path.join(GM,  'GMtrain.npy'))
#Xte = np.load(os.path.join(GM,  'GMtest.npy'))
#ytr = np.load(os.path.join(GM,  'ytrain.npy'))
enettv.fit(Xgm_train, y_train)
p_lr_l2.fit(Xroi_train, pop_train.Label.values)

arr = np.zeros(mask_im.shape)
arr[mask_im.get_data() != 0] = enettv.beta[penalty_start:].ravel()
out_im = nibabel.Nifti1Image(arr, affine=mask_im.get_affine())
out_im.to_filename(os.path.join(OUTPUT, "beta.nii"))
np.save(os.path.join(OUTPUT, "beta.npy"), enettv.beta)


test = pd.DataFrame(np.hstack([
pop_test.SID[:, np.newaxis],
pop_test.Label[:, np.newaxis],
p_lr_l2.predict_proba(Xroi_test)[:, [1]],
p_lr_l2.predict(Xroi_test)[:, np.newaxis],
enettv.predict_probability(Xgm_test),
enettv.predict(Xgm_test)]),
columns = ["SID", "true", "prob_roi", "pred_roi", "prob_gm", "pred_gm"])
#test.true = test.true.astype(int)
test.pred_gm = test.pred_gm.astype(int)
test.pred_roi = test.pred_roi.astype(int)


train = pd.DataFrame(np.hstack([
pop_train.SID[:, np.newaxis],
pop_train.Label[:, np.newaxis],
p_lr_l2.predict_proba(Xroi_train)[:, [1]],
p_lr_l2.predict(Xroi_train)[:, np.newaxis],
enettv.predict_probability(Xgm_train),
enettv.predict(Xgm_train)]),
columns = ["SID", "true", "prob_roi", "pred_roi", "prob_gm", "pred_gm"])
train.true = train.true.astype(int)
train.pred_gm = train.pred_gm.astype(int)
train.pred_roi = train.pred_roi.astype(int)


# COMBINE
test["prob_mix_mean"] = (test.prob_roi + test.prob_gm) / 2
test["pred_mix_mean"] = (test.prob_mix_mean > 0.5).astype(int)
train["prob_mix_mean"] = (train.prob_roi + train.prob_gm) / 2
train["pred_mix_mean"] = (train.prob_mix_mean > 0.5).astype(int)

print test[1:10]
print train[1:10]

train.to_csv(os.path.join(OUTPUT, "train.csv"), index=True)
test.to_csv(os.path.join(OUTPUT, "test.csv"), index=True)

test = pd.read_csv(os.path.join(OUTPUT, "test.csv"))
train = pd.read_csv(os.path.join(OUTPUT, "train.csv"))

# Compute train scores
acc_gm_mean_train = accuracy_score(train.true, train.pred_gm.values)
fpr, tpr, thcv_testholds = roc_curve(train.true, train.pred_gm)
roc_auc_gm_train = auc(fpr, tpr)

acc_roi_train = accuracy_score(train.true, train.pred_roi.values)
fpr, tpr, thcv_testholds = roc_curve(train.true, train.pred_roi)
roc_auc_roi_train = auc(fpr, tpr)

acc_mix_mean_train = accuracy_score(train.true, train.pred_mix_mean.values)
fpr, tpr, thcv_testholds = roc_curve(train.true, train.pred_mix_mean)
roc_auc_mix_mean_train = auc(fpr, tpr)


train_summary = pd.DataFrame([
["gm", acc_gm_mean_train, roc_auc_gm_train],
["roi", acc_roi_train, roc_auc_roi_train],
["mix", acc_mix_mean_train, roc_auc_mix_mean_train]], columns = ["featues", "acc", "auc"])

print "Train summary"
print train_summary

print "Test count gm", np.bincount(test.pred_gm)
print "Test count roi", np.bincount(test.pred_roi)
print "Test count mix", np.bincount(test.pred_mix_mean)
