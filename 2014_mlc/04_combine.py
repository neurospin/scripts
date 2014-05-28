# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:20:11 2014

@author: ed203246
"""

import os.path, sys
import gzip
import numpy as np
import pandas as pd
import json
import pylab as pl
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper

IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")
INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2014_mlc")
sys.path.append(SRC_PATH)
import utils



GM = "/neurospin/brainomics/2014_mlc/GM"
WHICH = "0.05_0.45_0.45_0.1"
penalty_start = 1
#/neurospin/brainomics/2014_mlc/GM/results/0/0.05_0.45_0.45_0.1/
pop_train = pd.read_csv(INPUT_SUBJECT_TRAIN)

cv = json.load(open(os.path.join(GM, "config.json")))['resample']
#cv = StratifiedKFold(y_train, n_folds=5)
Xgm_train = np.load(os.path.join(GM,  'GMtrain.npy'))
Xroi_train = pd.read_csv(INPUT_ROI_TRAIN, header=None).values

# enettv for GM
alpha, l1, l2, tv = [float(p) for p in WHICH.split("_")]
l1, l2, tv = alpha * l1, alpha * l2, alpha * tv
enettv = LogisticRegressionL1L2TV(l1, l2, tv, 0, penalty_start=penalty_start,
                                   class_weight="auto")

# lr l2 for roi
p_lr_l2 = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(C=0.005, penalty='l2')),
])

###############################################################################
# LOAD W check we retrive same results

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
    y_pred_replay.append(enettv.predict(Xgm_train[te,: ]).ravel())
    W.append(w.ravel())
    y_true_disk.append(np.load(os.path.join(input_path, "y_true.npy")).ravel())
    y_pred_disk.append(np.load(os.path.join(input_path, "y_pred.npy")).ravel())
    y_true_csv.append(pop_train.Label.values[te])

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


###############################################################################
## RUN separatly
res = list()
subjects = list()
for i, (tr, te) in enumerate(cv):
    #i, tr, te = 0, cv[0][0], cv[0][1]
    #Fit
    p_lr_l2.fit(Xroi_train[tr, :], pop_train.Label.values[tr])
    enettv.beta = W[i, :][:, np.newaxis]
    # Predict
    subjects.append(pop_train.SID.values[te])
    res.append(np.hstack([
    pop_train.Label.values[te][:, np.newaxis],
    p_lr_l2.predict_proba(Xroi_train[te,: ])[:, [1]],
    p_lr_l2.predict(Xroi_train[te,: ])[:, np.newaxis],
    enettv.predict_probability(Xgm_train[te,: ]),
    enettv.predict(Xgm_train[te,: ])]))

res = np.vstack(res)
subjects = np.hstack(subjects)

res= pd.DataFrame(res, index=subjects,
             columns=["y_true", "prob_roi", "pred_roi", "prob_gm", "pred_gm"])

# acc
acc_roi = accuracy_score(res.y_true, res.pred_roi)
acc_gm = accuracy_score(res.y_true, res.pred_gm)
# auc
fpr, tpr, thresholds = roc_curve(res.y_true, res.prob_roi)
roc_auc_roi = auc(fpr, tpr)
fpr, tpr, thresholds = roc_curve(res.y_true, res.prob_gm)
roc_auc_gm = auc(fpr, tpr)
# recalls
_, r_roi, _, _ = precision_recall_fscore_support(res.y_true, res.pred_roi, average=None)
_, r_gm, _, _ = precision_recall_fscore_support(res.y_true, res.pred_gm, average=None)

summary = pd.DataFrame([
["roi"] + [acc_roi] + [roc_auc_roi] + r_roi.tolist(),
["gm"]  + [acc_gm] + [roc_auc_gm] + r_gm.tolist()],
columns=["features", "acc", "auc", "recall_0", "recall_1"])

print "Individuals classifiers performances"
print "===================================="
print summary
#  feat       acc       auc  recall_0  recall_1
#0  roi  0.686667  0.765511  0.720000  0.653333
#1   gm  0.680000  0.708622  0.693333  0.666667
#
# Compare
same = np.sum(res.pred_roi == res.pred_gm) / float(len(res.pred_roi))
assert same == 0.6333333333333333
# Mc nemar test
mcnemar_pval, cont_table = utils.mcnemar_test_prediction(y_pred1=res.pred_roi, y_pred2=res.pred_gm,
                              y_true=res.y_true, cont_table=True)
assert mcnemar_pval == 1
# No differences
print "Contingency table"
print cont_table
#       2_Pos  1_Neg  Tot
#1_Pos     75     28  103
#1_Neg     27     20   47
#Tot      102     48  150
print "Mcnemar pval=", mcnemar_pval

pl.plot(res.prob_roi[res.y_true==0], res.prob_gm[res.y_true==0], "bo",
         res.prob_roi[res.y_true==1], res.prob_gm[res.y_true==1], "ro")
pl.plot([0.5, 0.5], [0, 1], 'k-')
pl.plot([0, 1], [0.5, 0.5], 'k-')
pl.xlabel('Prob(1|ROI)')
pl.ylabel('Prob(1|GM)')
pl.title('Posterior probas (Blue: 0, Red:1)')
pl.show()

#############################################################
# Consensus classif
# Mean
res.prob_mix_mean = (res.prob_roi + res.prob_gm) / 2
res.pred_mix_mean = (res.prob_mix_mean > 0.5).astype(int)
_, r_mix_mean, _, _ = precision_recall_fscore_support(res.y_true, res.pred_mix_mean, average=None)
acc_mix_mean = accuracy_score(res.y_true, res.pred_mix_mean)
fpr, tpr, thresholds = roc_curve(res.y_true, res.pred_mix_mean)
roc_auc_mix_mean = auc(fpr, tpr)

# prod
res.prob_mix_prod = (res.prob_roi * res.prob_gm)
res.pred_mix_prod = (res.prob_mix_prod > 0.25).astype(int)
_, r_mix_prod, _, _ = precision_recall_fscore_support(res.y_true, res.pred_mix_prod, average=None)
acc_mix_prod = accuracy_score(res.y_true, res.pred_mix_prod)
fpr, tpr, thresholds = roc_curve(res.y_true, res.pred_mix_prod)
roc_auc_mix_prod = auc(fpr, tpr)

# Max
prob_both = np.vstack([res.prob_roi, res.prob_gm]).T
res.prob_mix_max = np.max(prob_both, axis=1)
res.pred_mix_max = (res.prob_mix_max > 0.5).astype(int)
_, r_mix_max, _, _ = precision_recall_fscore_support(res.y_true, res.pred_mix_max, average=None)
acc_mix_max = accuracy_score(res.y_true, res.pred_mix_max)
fpr, tpr, thresholds = roc_curve(res.y_true, res.pred_mix_max)
roc_auc_mix_max = auc(fpr, tpr)

# Learn lr
s = StandardScaler()
x = s.fit_transform(prob_both)
lr = LogisticRegression()
lr.fit(x, res.y_true)
res.prob_mix_learn = lr.predict_proba(x)[:, 1]
res.pred_mix_learn = lr.predict(x)
_, r_mix_learn, _, _ = precision_recall_fscore_support(res.y_true, res.pred_mix_learn, average=None)
acc_mix_learn = accuracy_score(res.y_true, res.pred_mix_learn)
fpr, tpr, thresholds = roc_curve(res.y_true, res.pred_mix_learn)
roc_auc_mix_learn = auc(fpr, tpr)

print "Mixer classifiers performances"
print "=============================="

summary_mix = pd.DataFrame([
["Mean"] + [acc_mix_mean] + [roc_auc_mix_mean] + r_mix_mean.tolist(),
["max"] + [acc_mix_max] + [roc_auc_mix_max] + r_mix_max.tolist(),
["prod"] + [acc_mix_prod] + [roc_auc_mix_prod] + r_mix_prod.tolist(),
["learn"] + [acc_mix_learn] + [roc_auc_mix_learn] + r_mix_learn.tolist()],
columns=["Mixer", "acc", "auc", "recall_0", "recall_1"])

print summary_mix