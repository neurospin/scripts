# -*- coding: utf-8 -*-
"""
Created on Tue May 27 18:20:11 2014

@author: ed203246
run scripts/2014_mlc/04_combine.py
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
import nibabel

IMAGES_PATH = "/neurospin/tmp/mlc2014/processed/binary"
INPUT_ROI_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_data.csv")
INPUT_SUBJECT_TRAIN = os.path.join(IMAGES_PATH, "BinaryTrain_sbj_list.csv")
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2014_mlc")
sys.path.append(SRC_PATH)
import utils
"""
run /home/ed203246/git/scripts/2014_mlc/04_combine_cv.py
"""

#CV = "5CV"
CV = "10CV"

# 5CV
if CV == "5CV":
    GM = "/neurospin/brainomics/2014_mlc/GM"
    WHICH = "0.05_0.45_0.45_0.1"
    print """
    gm: Grey Matter map processed with Elasticnet TV logistic regression. Maximize:
    Logistic regression with L1, L2 and TV penalties:
            f(beta) = - loglik/n_samples +
                      + l1 * ||beta||_1
                      + (l2 / 2 * n) * ||beta||²_2
                      + tv * TV(beta)
    """
    print "with parameters l1=%.3f, l2=%.3f, tv=%.3f, found by CV" % (l1, l2, tv)
    
    print
    print "roi: preprocessed features processed with Elasticnet TV logistic regression"
    print "with parameters  C=%.3f" % C
    #WHICH = "0.05_0.08_0.72_0.2"

if CV == "10CV":
    GM = "/neurospin/brainomics/2014_mlc/GM_10CV"
    WHICH = "0.05_0.45_0.45_0.1_-1.0"
    #WHICH = "0.05_0.08_0.72_0.2_-1.0"

penalty_start = 1
#/neurospin/brainomics/2014_mlc/GM/results/0/0.05_0.45_0.45_0.1/
pop_train = pd.read_csv(INPUT_SUBJECT_TRAIN)

cv = json.load(open(os.path.join(GM, "config.json")))['resample']
#cv = StratifiedKFold(y_train, n_folds=5)
Xgm_train = np.load(os.path.join(GM,  'GMtrain.npy'))
Xgm_test = np.load(os.path.join(GM,  'GMtest.npy'))
y_train = np.load(os.path.join(GM,  'ytrain.npy'))
assert np.all(pop_train.Label.values  == y_train.ravel())

Xroi_train = pd.read_csv(INPUT_ROI_TRAIN, header=None).values

# enettv for GM
arg = [float(p) for p in WHICH.split("_")]
if len(arg) == 4:
    alpha, l1, l2, tv = arg
else:
    alpha, l1, l2, tv, k = arg


l1, l2, tv = alpha * l1, alpha * l2, alpha * tv
enettv = LogisticRegressionL1L2TV(l1, l2, tv, 0, penalty_start=penalty_start,
                                   class_weight="auto")
C = 0.0022
# lr l2 for roi
p_lr_l2 = Pipeline([
    ('scaler', StandardScaler()),
   # ('classifier', LogisticRegression(C=0.005, penalty='l2')),
    ('classifier', LogisticRegression(C=C, penalty='l2')),
])


print "=========="
print "== %s ==" % CV
print "=========="


#print "enettv", WHICH, GM

###############################################################################
# LOAD W check we retrive same results

W = list()
y_true_disk = list()
y_true_csv = list()
#y_true_train = list()
#y_pred_train = list()
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
    #y_true_train.append(pop_train.Label.values[tr])
    #y_pred_train.append(enettv.predict(Xgm_train[tr,: ]).ravel())

W = np.vstack(W)
y_true_disk = np.hstack(y_true_disk)
y_true_csv = np.hstack(y_true_csv)
y_pred_disk = np.hstack(y_pred_disk)
y_pred_replay = np.hstack(y_pred_replay)
#y_true_train = np.hstack(y_true_train)
#y_pred_train = np.hstack(y_pred_train)

# Do some QC
assert np.all(y_true_disk == y_true_csv)  # true test == those in csv
assert np.all(y_pred_replay == y_pred_disk) # pred == those stored

p, r, f, s = precision_recall_fscore_support(y_true_csv, y_pred_replay, average=None)
#assert r.mean() == 0.67999999999999994
#_, cv_train_recall, _, _ = precision_recall_fscore_support(y_true_train, y_pred_train, average=None)

###############################################################################
## RUN separatly
print "------------------------------------------"
print "-- Individuals classifiers performances --" 
print "------------------------------------------"
cv_test = list()
cv_train = list()
subjects_train = list()
subjects_test = list()
for i, (tr, te) in enumerate(cv):
    #i, tr, te = 0, cv[0][0], cv[0][1]
    #Fit
    p_lr_l2.fit(Xroi_train[tr, :], pop_train.Label.values[tr])
    enettv.beta = W[i, :][:, np.newaxis]
    # Predict
    subjects_test.append(pop_train.SID.values[te])
    cv_test.append(np.hstack([
        pop_train.Label.values[te][:, np.newaxis],
        p_lr_l2.predict_proba(Xroi_train[te,: ])[:, [1]],
        p_lr_l2.predict(Xroi_train[te,: ])[:, np.newaxis],
        enettv.predict_probability(Xgm_train[te,: ]),
        enettv.predict(Xgm_train[te,: ])]))
    subjects_train.append(pop_train.SID.values[tr])
    cv_train.append(np.hstack([
        pop_train.Label.values[tr][:, np.newaxis],
        p_lr_l2.predict_proba(Xroi_train[tr,: ])[:, [1]],
        p_lr_l2.predict(Xroi_train[tr,: ])[:, np.newaxis],
        enettv.predict_probability(Xgm_train[tr,: ]),
        enettv.predict(Xgm_train[tr,: ])]))


print "~~ TEST performances ~~"
print "~~~~~~~~~~~~~~~~~~~~~~~"
cv_test = np.vstack(cv_test)
subjects_test = np.hstack(subjects_test)

cv_test= pd.DataFrame(cv_test, index=subjects_test,
             columns=["y_true", "prob_roi", "pred_roi", "prob_gm", "pred_gm"])

# acc
acc_roi = accuracy_score(cv_test.y_true, cv_test.pred_roi)
acc_gm = accuracy_score(cv_test.y_true, cv_test.pred_gm)
# auc
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.prob_roi)
roc_auc_roi = auc(fpr, tpr)
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.prob_gm)
roc_auc_gm = auc(fpr, tpr)
# recalls
_, r_roi, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_roi, average=None)
_, r_gm, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_gm, average=None)

summary_test_cv = pd.DataFrame([
["roi"] + [acc_roi] + [roc_auc_roi] + r_roi.tolist(),
["gm"]  + [acc_gm] + [roc_auc_gm] + r_gm.tolist()],
columns=["features", "acc", "auc", "recall_0", "recall_1"])

print summary_test_cv

print "~~ Comparison ~~"
print "~~~~~~~~~~~~~~~~~~"
print "Cross-classification contingency table c1=gm, c2=roi, c1_0: gm predict 1"
cross_classif_test = utils.contingency_table(c1=cv_test.pred_gm, c2=cv_test.pred_roi)
print cross_classif_test
print "Classifier agreed %.2f" % (np.sum(cv_test.pred_roi == cv_test.pred_gm) / float(len(cv_test.pred_roi)))
#assert same == 0.6333333333333333
# Mc nemar test
mcnemar_pval, cont_table = utils.mcnemar_test_prediction(y_pred1=cv_test.pred_roi, y_pred2=cv_test.pred_gm,
                              y_true=cv_test.y_true, cont_table=True)
#assert mcnemar_pval == 1
# No differences
print "McNemmar contingency table"
print cont_table
#       2_Pos  1_Neg  Tot
#1_Pos     75     28  103
#1_Neg     27     20   47
#Tot      102     48  150
print "Mcnemar pval=", mcnemar_pval

pl.plot(cv_test.prob_roi[cv_test.y_true==0], cv_test.prob_gm[cv_test.y_true==0], "bo",
         cv_test.prob_roi[cv_test.y_true==1], cv_test.prob_gm[cv_test.y_true==1], "ro")
pl.plot([0.5, 0.5], [0, 1], 'k-')
pl.plot([0, 1], [0.5, 0.5], 'k-')
pl.xlabel('Prob(1|ROI)')
pl.ylabel('Prob(1|GM)')
pl.title('Posterior probas (Blue: 0, Red:1)')
pl.show()


print "~~ TRAIN performances ~~"
print "~~~~~~~~~~~~~~~~~~~~~~~~"
cv_train = np.vstack(cv_train)
subjects_train = np.hstack(subjects_train)

cv_train= pd.DataFrame(cv_train, index=subjects_train,
             columns=["y_true", "prob_roi", "pred_roi", "prob_gm", "pred_gm"])

# acc
acc_roi = accuracy_score(cv_train.y_true, cv_train.pred_roi)
acc_gm = accuracy_score(cv_train.y_true, cv_train.pred_gm)
# auc
fpr, tpr, thcv_trainholds = roc_curve(cv_train.y_true, cv_train.prob_roi)
roc_auc_roi = auc(fpr, tpr)
fpr, tpr, thcv_trainholds = roc_curve(cv_train.y_true, cv_train.prob_gm)
roc_auc_gm = auc(fpr, tpr)
# recalls
_, r_roi, _, _ = precision_recall_fscore_support(cv_train.y_true, cv_train.pred_roi, average=None)
_, r_gm, _, _ = precision_recall_fscore_support(cv_train.y_true, cv_train.pred_gm, average=None)

summary_train_cv = pd.DataFrame([
["roi"] + [acc_roi] + [roc_auc_roi] + r_roi.tolist(),
["gm"]  + [acc_gm] + [roc_auc_gm] + r_gm.tolist()],
columns=["features", "acc", "auc", "recall_0", "recall_1"])

print summary_train_cv

#print "Cross-classification contingency table c1=gm, c2=roi, c1_0: gm predict 1"
#cross_classif_test = utils.contingency_table(c1=cv_train.pred_gm, c2=cv_train.pred_roi)
#print cross_classif_test
#print "Classifier agreed %.2f" % (np.sum(cv_train.pred_roi == cv_train.pred_gm) / float(len(cv_test.pred_roi)))

#############################################################
# Consensus classif
# Mean
cv_test.prob_mix_mean = (cv_test.prob_roi + cv_test.prob_gm) / 2
cv_test.pred_mix_mean = (cv_test.prob_mix_mean > 0.5).astype(int)
_, r_mix_mean, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_mix_mean, average=None)
acc_mix_mean = accuracy_score(cv_test.y_true, cv_test.pred_mix_mean)
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.pred_mix_mean)
roc_auc_mix_mean = auc(fpr, tpr)

# prod
cv_test.prob_mix_prod = (cv_test.prob_roi * cv_test.prob_gm)
cv_test.pred_mix_prod = (cv_test.prob_mix_prod > 0.25).astype(int)
_, r_mix_prod, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_mix_prod, average=None)
acc_mix_prod = accuracy_score(cv_test.y_true, cv_test.pred_mix_prod)
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.pred_mix_prod)
roc_auc_mix_prod = auc(fpr, tpr)

# Max
prob_both = np.vstack([cv_test.prob_roi, cv_test.prob_gm]).T
cv_test.prob_mix_max = np.max(prob_both, axis=1)
cv_test.pred_mix_max = (cv_test.prob_mix_max > 0.5).astype(int)
_, r_mix_max, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_mix_max, average=None)
acc_mix_max = accuracy_score(cv_test.y_true, cv_test.pred_mix_max)
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.pred_mix_max)
roc_auc_mix_max = auc(fpr, tpr)

# Learn lr
s = StandardScaler()
x = s.fit_transform(prob_both)
lr = LogisticRegression()
lr.fit(x, cv_test.y_true)
cv_test.prob_mix_learn = lr.predict_proba(x)[:, 1]
cv_test.pred_mix_learn = lr.predict(x)
_, r_mix_learn, _, _ = precision_recall_fscore_support(cv_test.y_true, cv_test.pred_mix_learn, average=None)
acc_mix_learn = accuracy_score(cv_test.y_true, cv_test.pred_mix_learn)
fpr, tpr, thcv_testholds = roc_curve(cv_test.y_true, cv_test.pred_mix_learn)
roc_auc_mix_learn = auc(fpr, tpr)

print "--------------------------------------"
print "-- Combined classifiers performances --"
print "--------------------------------------"

summary_mix = pd.DataFrame([
["Mean"] + [acc_mix_mean] + [roc_auc_mix_mean] + r_mix_mean.tolist(),
["max"] + [acc_mix_max] + [roc_auc_mix_max] + r_mix_max.tolist(),
["prod"] + [acc_mix_prod] + [roc_auc_mix_prod] + r_mix_prod.tolist(),
["learn"] + [acc_mix_learn] + [roc_auc_mix_learn] + r_mix_learn.tolist()],
columns=["Mixer", "acc", "auc", "recall_0", "recall_1"])

print summary_mix

