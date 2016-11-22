# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:50:17 2016

@author: ad247405
"""



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


BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
INPUT_CSV = os.path.join(BASE_PATH,"population26oct.txt")
pop = pd.read_csv(INPUT_CSV,delimiter=' ')

#out_scores_file = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/without_subject19/svm_with_reprocessRS_no_model_selection_without_subject19/scores.txt'
out_scores_file = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results26/multivariate_analysis/svm/with_RS_no_model_selection/scores.txt'

##################################################################################
mask = nibabel.load('/neurospin/brainomics/2016_classif_hallu_fmri_bis/atlases/MNI152_T1_3mm_brain_mask.nii.gz')
mask_bool = mask.get_data() !=0

number_features = mask_bool.sum()
number_subjects = pop.shape[0]
#############################################################################



 #Retreive variables
#############################################################################

T= np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','T.npy'))
betas = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','betas.npy'))
y_state = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','y_state.npy'))
subject = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','subject.npy'))

T_RS = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','T_RS.npy'))
y_state_RS = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','y_RS.npy'))
subject_RS = np.load(os.path.join(BASE_PATH,'results26','multivariate_analysis','data','subject_RS.npy'))


T = np.nan_to_num(T)

T_RS = np.nan_to_num(T_RS)


Tdiff=np.mean(T_RS,axis=0)-np.mean(T[y_state==0],axis=0)
T_RS_diff=T_RS-Tdiff





#SVM & Leave one subject-out - no feature selection
#############################################################################
outf=open(out_scores_file, "w")
outf.write("C"+" "+"accuracy"+" "+"sensitivity"+" "+"specificity"+" "+"weights_map_path\n")
outf.flush()


C_range = [100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]


for c in C_range:   
    n=0
    list_predict=list()
    list_true=list()
    coef=np.zeros((number_subjects,number_features))
    clf = svm.LinearSVC(C=c,fit_intercept=False,class_weight='auto')
    for i in range(0,number_subjects):
        if i !=19:
            test_bool=(subject==i)
            train_bool=(subject!=i)
            Xtest=T[test_bool,:]
            ytest=y_state[test_bool]
            Xtrain=np.vstack((T_RS_diff,T[train_bool,:]))
            ytrain=np.hstack((y_state_RS,y_state[train_bool]))
            list_true.append(ytest.ravel())
            scaler = preprocessing.StandardScaler().fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest=scaler.transform(Xtest)
            clf.fit(Xtrain, ytrain.ravel())
            coef[n,:]=clf.coef_
            pred=(clf.predict(Xtest))
            list_predict.append(pred)
            print n 
            n=n+1 
        
    #############################################################################
    #Save weights and std of SVM coef
    std_coef=coef.std(axis=0)
    mean_coef=coef.mean(axis=0)
    arr = np.zeros(mask_bool.shape);
    arr[mask_bool] = mean_coef
    out_im = nibabel.Nifti1Image(arr, affine=mask.get_affine())
    filename = os.path.join(BASE_PATH,"results26","multivariate_analysis","svm","with_RS_no_model_selection","coef_mean_c=%r.nii.gz")%c

    out_im.to_filename(filename)
        
     #Display scores of classification (accuracy, precision and recall)
    #############################################################################   
    
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    acc=metrics.accuracy_score(t,p)
    pre=recall_scores[0]
    rec=recall_scores[1]
    print acc
    print pre
    print rec
#############################################################################
#############################################################################   

    outf.write(str(c)+" "+ str(acc)+" "+str(pre)+" "+str(rec)+" "+ filename+"\n")

    outf.flush()

outf.close()



