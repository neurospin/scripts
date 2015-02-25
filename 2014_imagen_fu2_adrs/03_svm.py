# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 10:26:51 2015

@author: cp243490
"""

from sklearn import svm
from sklearn.metrics import recall_score
import nibabel as nib
import numpy as np
import json
import os

BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs"

DATASET_PATH = os.path.join(BASE_PATH,    "ADRS_datasets")

ENETTV_PATH = os.path.join(BASE_PATH, "ADRS_enettv")

SVM_PATH = os.path.join(BASE_PATH, "ADRS_svm")
if not os.path.exists(SVM_PATH):
    os.makedirs(SVM_PATH)
output_file = os.path.join(SVM_PATH, 'svm_scores.csv')

#penalty_start = 3

n_folds = 5

INPUT_MASK = os.path.join(DATASET_PATH, 'mask_atlas_binarized.nii.gz')
mask = nib.load(INPUT_MASK).get_data()
INPUT_X = os.path.join(DATASET_PATH, 'X.npy')
X = np.load(INPUT_X, mmap_mode='r')
INPUT_y = os.path.join(DATASET_PATH, 'y.npy')
y = np.load(INPUT_y)
y = y.ravel()
#########################################################################
## LOAD config file
config_file = os.path.join(ENETTV_PATH,
                                     "config.json")
# open config file
config = json.load(open(config_file))
# get resample index for each outer
resample = config["resample"]
resample = np.asarray(resample)
resample = resample[1:]
y_true = []
y_pred = []
for j in xrange(len(resample)):
    cv = resample[j]
    if cv is not None:
        y_train = y[cv[0], :]
        X_test = X[cv[1], :]
        y_test = y[cv[1], :]
        X_train = X[cv[0], :]
    else:
        y_test = y_train = y
        X_test = X_train = X

    svmlin = svm.LinearSVC(fit_intercept=False, class_weight='auto')
    svmlin.fit(X_train, y_train)
    y_pred.append(svmlin.predict(X_test))
    y_true.append(y_test)
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)
# global accuracy
accuracy = np.sum(y_pred == y_true) / float(len(y_true))
print accuracy

# Use Scorer
# scorer = Scorer(score_func=recall_score, pos_label=None, average='macro')
# scorer(estimator=svmlin, X=X_test, y=y_test)

# confusion_matrix(y_true=y_test, y_pred=y_pred, labels=None)
recall_scores = recall_score(y_true=y_true, y_pred=y_pred,
                             pos_label=None,
                             average=None, labels=[0, 1])
bsr = recall_scores.mean()
print recall_scores, bsr
#print "save results of the inner cross-validation : ", SVM_PATH
#scores_tab.to_csv(output_file, index=False)