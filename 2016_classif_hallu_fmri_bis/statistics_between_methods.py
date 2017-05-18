#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:47:46 2016

@author: ad247405
"""


import pandas as pd
import scipy
from scipy import stats

#This script is used to run statistical test to assess difference of performance 
#betwen SVM and ENETTV

results_svm_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/results_dCV.xlsx"
results_enettv_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/results_dCV.xlsx"



# TRANS vs OFF
results_svm_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/model_selection/results_dCV.xlsx"
results_enettv_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/model_selection/results_dCV.xlsx"

svm_scores_argmax_byfold = pd.read_excel(results_svm_path,sheetname = "scores_argmax_byfold")
enettv_scores_argmax_byfold = pd.read_excel(results_enettv_path,sheetname = "scores_argmax_byfold")

T,p = scipy.stats.ttest_1samp(enettv_scores_argmax_byfold.auc - svm_scores_argmax_byfold.auc,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher AUC than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher AUC than EnetTV")    
#else:
#    print ("No significantl differences in AUC between EnetTV and SVM")
   


# TRANS vs OFF, with IMA
results_svm_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_IMA_model_selection/results_dCV.xlsx"
results_enettv_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_IMA_model_selection/results_dCV.xlsx"

svm_scores_argmax_byfold = pd.read_excel(results_svm_path,sheetname = "scores_argmax_byfold")
enettv_scores_argmax_byfold = pd.read_excel(results_enettv_path,sheetname = "scores_argmax_byfold")

T,p = scipy.stats.ttest_1samp(enettv_scores_argmax_byfold.auc - svm_scores_argmax_byfold.auc,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher AUC than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher AUC than EnetTV")    
else:
    print ("No significantl differences in AUC between EnetTV and SVM")

    
    # TRANS vs OFF, with RS
results_svm_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_RS_model_selection/results_dCV.xlsx"
results_enettv_path = "/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_RS_model_selection/results_dCV.xlsx"

svm_scores_argmax_byfold = pd.read_excel(results_svm_path,sheetname = "scores_argmax_byfold")
enettv_scores_argmax_byfold = pd.read_excel(results_enettv_path,sheetname = "scores_argmax_byfold")

T,p = scipy.stats.ttest_1samp(enettv_scores_argmax_byfold.auc - svm_scores_argmax_byfold.auc,0.0)
print ("T_stat =  %s " %T)
print ("pvalue =  %s " %p)
if p<0.05 and T>0:
    print ("Enet TV reveals significantly higher AUC than SVM")
if p<0.05 and T<0:
    print ("SVM reveals significantly higher AUC than EnetTV")    
else:
    print ("No significantl differences in AUC between EnetTV and SVM")

