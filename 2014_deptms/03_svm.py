# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 16:58:35 2015

@author: cp243490
"""

from sklearn import svm
from sklearn.metrics import recall_score
import nibabel as nib
import numpy as np
import json
import os
import pandas as pd
from collections import OrderedDict

BASE_PATH = "/neurospin/brainomics/2014_deptms"

MOD = "MRI"

DATASET_PATH = os.path.join(BASE_PATH,    "datasets", MOD)

INPUT_ROIS_CSV = os.path.join(BASE_PATH, "base_data", "ROI_labels.csv")

ENETTV_PATH = os.path.join(BASE_PATH, "results_enettv")

SVM_PATH = os.path.join(BASE_PATH, "results_svm")
if not os.path.exists(SVM_PATH):
    os.makedirs(SVM_PATH)
output_file = os.path.join(SVM_PATH, 'svm_scores.csv')

#penalty_start = 3

n_folds = 10
#############################################################################
## Read ROIs csv
rois = []
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["atlas_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if ((not roi_name in rois)
          and (roi_name != "Maskdep-sub") and (roi_name != "Maskdep-cort")):
            print "ROI: ", roi_name
            rois.append(roi_name)
rois.append("brain")
print "\n"

for i, roi in enumerate(rois):
    print "ROI", roi
    scores_svm = OrderedDict()
    scores_svm['ROI'] = roi
    INPUT_MASK = os.path.join(DATASET_PATH, 'mask_' + roi + '.nii')
    mask = nib.load(INPUT_MASK).get_data()
    INPUT_X = os.path.join(DATASET_PATH, 'X_' + MOD + '_' + roi + '.npy')
    X = np.load(INPUT_X)
    INPUT_y = os.path.join(DATASET_PATH, 'y.npy')
    y = np.load(INPUT_y)
    y = y.ravel()
    #########################################################################
    ## LOAD config file
    config_selection_file = os.path.join(ENETTV_PATH, MOD + '_' + roi,
                                         "config_dCV_validation.json")
    # open config file
    config_selection = json.load(open(config_selection_file))
    # get resample index for each outer
    resample = config_selection["resample"]
    resample = np.asarray(resample)
    resample = resample[1:]
    y_true = []
    y_pred = []
    for j in xrange(len(resample)):
        cv = resample[j]
        if cv is not None:
            X_test = X[cv[1], ...]
            y_test = y[cv[1], ...]
            X_train = X[cv[2], ...]
            y_train = y[cv[2], ...]
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
    scores_svm['accuracy'] = accuracy
    scores_svm['specificity'] = recall_scores[0]
    scores_svm['sensitivity'] = recall_scores[1]
    scores_svm['bsr'] = bsr
    if i == 0:
        scores_tab = pd.DataFrame(columns=scores_svm.keys())
    scores_tab.loc[i, ] = scores_svm.values()
    i += 1
    print roi, recall_scores, bsr
print "save results of the inner cross-validation : ", SVM_PATH
scores_tab.to_csv(output_file, index=False)