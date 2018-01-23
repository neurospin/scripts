#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:08:33 2017

@author: ad247405
"""

import numpy as np
import os
import re
import glob
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn import grid_search
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import random
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.metrics import recall_score
from sklearn import svm, metrics, linear_model
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import  make_scorer,accuracy_score,recall_score,precision_score


##############################################################################
U_all = np.load("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/U_scores_corrected/U_all.npy")
y_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/y.npy")
X_all = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/X.npy")
pop_all = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/population.csv")

labels_all_scz = np.load("/neurospin/brainomics/2016_schizConnect/\
2018_analysis_2ndpart_clinic/results/clustering/corrected_results/\
correction_age_sex_site/2_clusters_solution/labels_cluster.npy")

age = pop_all["age"].values
sex = pop_all["sex_num"].values
df = pd.DataFrame()
df["site"] = pop_all["site_num"].values
df["labels"] = np.nan
df["labels"][y_all==1] = labels_all_scz
df["labels"][y_all==0] = "controls"
LABELS_DICT = {"controls":"controls",0: "cluster 1", 1: "cluster 2"}
df["labels_name"]  = df["labels"].map(LABELS_DICT)

labels = df["labels"].values
labels_name = df["labels_name"].values


#1) CONTROLS vs CLUSTER 1
X = U_all[labels_name !="cluster 2",:]
#X = X_all[labels_name !="cluster 2",:]

y = y_all[labels_name !="cluster 2"]
assert sum(y==0) == sum(labels_name=="controls") == 330
assert sum(y==1) == sum(labels_name=="cluster 1") == 144
assert X.shape[0] == y.shape[0] == 474
bacc,recall = svm_score(X,y)
print("Balanced acc: %s, Sen: %s, Spe : %s"%(bacc,recall[0],recall[1]))
#Balanced acc: 0.759090909091, Sen: 0.684848484848, Spe : 0.833333333333

#1) CONTROLS vs SCZ
X = U_all
X = X_all
y = y_all
assert sum(y==0) == sum(labels_name=="controls") == 330
assert sum(y==1) == 276
assert X.shape[0] == y.shape[0] == 606
bacc,recall = svm_score(X,y)
print("Balanced acc: %s, Sen: %s, Spe : %s"%(bacc,recall[0],recall[1]))
#Balanced acc: 0.71442687747, Sen: 0.69696969697, Spe : 0.731884057971

#2) CONTROLS vs CLUSTER 2
X = U_all[labels_name !="cluster 1",:]
X = X_all[labels_name !="cluster 1",:]
y = y_all[labels_name !="cluster 1"]
assert sum(y==0) == sum(labels_name=="controls") == 330
assert sum(y==1) == sum(labels_name=="cluster 2") == 132
assert X.shape[0] == y.shape[0] == 462
bacc,recall = svm_score(X,y)
print("Balanced acc: %s, Sen: %s, Spe : %s"%(bacc,recall[0],recall[1]))
#Balanced acc: 0.702272727273, Sen: 0.639393939394, Spe : 0.765151515152

#3) CLUSTER 1 vs CLUSTER 2
X = U_all[labels_name !="controls",:]
X = X_all[labels_name !="controls",:]
y = labels[labels_name !="controls"].astype(np.float)
assert sum(y==0) == sum(labels_name=="cluster 1") == 144
assert sum(y==1) == sum(labels_name=="cluster 2") ==132
assert X.shape[0] == y.shape[0] == 276
bacc,recall = svm_score(X,y)
print("Balanced acc: %s, Sen: %s, Spe : %s"%(bacc,recall[0],recall[1]))
#Balanced acc: 0.757575757576, Sen: 0.75, Spe : 0.765151515152



##############################################################################

#1) CONTROLS vs SCZ
X = U_all
X = X_all
y = y_all
assert sum(y==0) == sum(labels_name=="controls") == 330
assert sum(y==1) == 276
assert X.shape[0] == y.shape[0] == 606
bacc,recall = svm_score(X,y)
print("Balanced acc: %s, Sen: %s, Spe : %s"%(bacc,recall[0],recall[1]))
#Balanced acc: 0.71442687747, Sen: 0.69696969697, Spe : 0.731884057971


np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/corrected_results/correction_age_sex_site/2_clusters_solution/svm/true.npy",t)
np.save("/neurospin/brainomics/2016_schizConnect/2018_analysis_2ndpart_clinic/results/clustering/corrected_results/correction_age_sex_site/2_clusters_solution/svm/pred.npy",p)

t==p
true_0 = t[t==0]
true_1 = t[t==1]
pred_true_0 = p[t==0]
pred_true_1 = p[t==1]
labels_true_1 = labels_name[t==1]
labels_true_0 = labels_name[t==0]

labels_true_1[pred_true_1==0]#74 SCZ patients have been predicted as controls

sum(labels_true_1[pred_true_1==0]=="cluster 1") #25 from cluster 1
sum(labels_true_1[pred_true_1==0]=="cluster 2") #42 from cluster 2



def svm_score(X,y):

    list_predict=list()
    list_true=list()

    clf= svm.LinearSVC(fit_intercept=False,class_weight='balanced')
    parameters={'C':[1,1e-1,1e1]}

    def balanced_acc(t, p):
        recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
        ba = recall_scores.mean()
        return ba


    score=make_scorer(balanced_acc,greater_is_better=True)
    clf = grid_search.GridSearchCV(clf,parameters,cv=3,scoring=score)
    skf = StratifiedKFold(y,5)

    for train, test in skf:

        X_train=X[train,:]
        X_test=X[test,:]
        y_train=y[train]
        y_test=y[test]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        list_predict.append(y_pred)

    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    balanced_acc= recall_scores.mean()
    return balanced_acc,recall_scores
