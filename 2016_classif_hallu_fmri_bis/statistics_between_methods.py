#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:47:46 2016

@author: ad247405
"""


import pandas as pd
import scipy
from scipy import stats
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, recall_score

#This script is used to run statistical test to assess difference of performance
#betwen SVM and ENETTV


y_true_svm = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/score_by_cv/y_true.npy")
y_pred_svm = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/score_by_cv/y_pred.npy")

recall_mean_vector_svm = np.zeros((37))
recall_0_vector_svm = np.zeros((37))
recall_1_vector_svm = np.zeros((37))
auc_vector_svm = np.zeros((37))

for i in range(37):
    p, r, f, s = precision_recall_fscore_support(y_true_svm[i], y_pred_svm[i], average=None)
    auc = roc_auc_score(y_true_svm[i], y_pred_svm[i])
    recall_mean_vector_svm[i] = r.mean()
    recall_0_vector_svm[i] = r[0]
    recall_1_vector_svm[i] = r[1]
    auc_vector_svm[i] = auc


y_true_enettv = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/score_by_cv/y_true.npy")
y_pred_enettv = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/score_by_cv/y_pred.npy")
y_prob_pred_enettv = np.load("/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/score_by_cv/prob_pred.npy")

recall_mean_vector_enettv = np.zeros((37))
recall_0_vector_enettv = np.zeros((37))
recall_1_vector_enettv = np.zeros((37))
auc_vector_enettv = np.zeros((37))

for i in range(37):
    p, r, f, s = precision_recall_fscore_support(y_true_enettv[i], y_pred_enettv[i], average=None)
    auc = roc_auc_score(y_true_enettv[i], y_prob_pred_enettv[i])
    recall_mean_vector_enettv[i] = r.mean()
    recall_0_vector_enettv[i] = r[0]
    recall_1_vector_enettv[i] = r[1]
    auc_vector_enettv[i] = auc




T,p = scipy.stats.ttest_1samp(auc_vector_enettv - auc_vector_svm,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher AUC than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher AUC than EnetTV")


T,p = scipy.stats.ttest_1samp(recall_mean_vector_enettv - recall_mean_vector_svm,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher Acc than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher Acc than EnetTV")


T,p = scipy.stats.ttest_1samp(recall_0_vector_enettv - recall_0_vector_svm,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher Spe than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher Spe than EnetTV")


T,p = scipy.stats.ttest_1samp(recall_1_vector_enettv - recall_1_vector_svm,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher Sen than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher Sen than EnetTV")





