# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:33:08 2016

@author: ad247405
"""

import sklearn
import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import os
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn import grid_search, metrics
import brainomics.image_atlas
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from mulm import MUOLS
import scipy
from scipy import stats 
from sklearn.preprocessing import StandardScaler      

BASE_PATH= "/neurospin/brainomics/2016_deptms"
INPUT_CSV = os.path.join(BASE_PATH,"population.csv")
mask_path =  "/neurospin/brainomics/2016_deptms/Freesurfer/data/mask.npy"
mask = np.load(mask_path)

OUTPUT = "/neurospin/brainomics/2016_deptms/Freesurfer/results/svm/svm_no_model_selection"
##################################################################################


#############################################################################
 #Retreive variables
#############################################################################
X = np.load('/neurospin/brainomics/2016_deptms/Freesurfer/data/X.npy')
y = np.load('/neurospin/brainomics/2016_deptms/Freesurfer/data/y.npy').ravel()
number_covariates = 2 #age + sex

out_scores_file = os.path.join(OUTPUT,'svm_scores.txt')
##########################
number_subjects = X.shape[0]
number_features = X.shape[1] -number_covariates
#SVM & Leave one subject-out - no feature selection
#############################################################################
outf=open(out_scores_file, "w")
outf.write("C"+" "+"accuracy"+" "+"recall_0"+" "+"recall_1"+" "+"weights_map_path\n")
outf.flush()


C_range = [100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
N_FOLDS = 10

for c in C_range:   
    n=0
    list_predict=list()
    list_true=list()
    coef=np.zeros((N_FOLDS,number_features))
    clf = svm.LinearSVC(C=c,fit_intercept=False,class_weight='auto')
    skf = StratifiedKFold(n_folds=N_FOLDS,y=y)
    for train_index, test_index in skf:
        X_test = X[test_index,:]
        X_train = X[train_index,:]
        y_test = y[test_index]
        y_train = y[train_index]
        list_true.append(y_test.ravel())
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test=scaler.transform(X_test)
        clf.fit(X_train, y_train.ravel())
        coef[n,:] = clf.coef_[0,number_covariates:]
        pred=(clf.predict(X_test))
        list_predict.append(pred)
        print n 
        n=n+1 


    #############################################################################
    #Save weights and std of SVM coef
    std_coef=coef.std(axis=0)
    mean_coef=coef.mean(axis=0)
    filename = os.path.join(OUTPUT,"coef_mean_c=%r.npy")%c
    np.save(filename,mean_coef)
        
     #Display scores of classification (accuracy, precision and recall)
    #############################################################################   
    
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    acc=metrics.accuracy_score(t,p)
    rec_0=recall_scores[0]
    rec_1=recall_scores[1]
    print acc
    print rec_0
    print rec_1
#############################################################################
#############################################################################   

    outf.write(str(c)+" "+ str(acc)+" "+str(rec_0)+" "+str(rec_1)+" "+ filename+"\n")

    outf.flush()

outf.close()
