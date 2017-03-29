# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:22:25 2015

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




# Read pop csv

#############################################################################
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
INPUT_CSV = os.path.join(BASE_PATH,"results", "patients.csv")
pop = pd.read_csv(os.path.join(BASE_PATH,"results", "patients.csv"))
periods = pd.read_csv(os.path.join(BASE_PATH,"results", "nperiods.csv"))
#############################################################################



#############################################################################
#Mask on resampled Images (We use intecept between Harvard/Oxford cort/sub mask and MNI152linT1 mask)
ref=os.path.join(BASE_PATH,"atlases","MNI152lin_T1_3mm_brain_mask.nii.gz")
babel_mask_atlas = brainomics.image_atlas.resample_atlas_harvard_oxford(ref=ref
,output=(os.path.join(BASE_PATH,"results","mask.nii.gz")),smooth_size=None,dilation_size=None)
a=babel_mask_atlas.get_data()
babel_mask=nibabel.load(ref)
b=babel_mask.get_data()
b[a==0]=0
mask_bool=b!=0
#############################################################################



#Retrieve variables
#############################################################################
T=np.load(os.path.join(BASE_PATH,'results','svm','T.npy'))
betas=np.load(os.path.join(BASE_PATH,'results','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'results','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'results','svm','subject.npy'))
#############################################################################
#############################################################################


#Plot histogram of pvalues 
#############################################################################
selector = SelectPercentile(f_classif, percentile=1)
selector.fit_transform(T,y.ravel())
hist(selector.pvalues_,10,normed=False)



#Permutation to check null hypothesis
yp=np.random.permutation(y)
selector.fit_transform(Tm,yp.ravel())
hist(selector.pvalues_,30,normed=False)
#############################################################################
#############################################################################

#SVM & Leave one subject-out - no feature selection
#############################################################################

n=0
list_predict=list()
list_true=list()
coef=np.zeros((24,63966))
#coef=np.zeros((24,8028))
clf = svm.LinearSVC(C=10e-7,fit_intercept=False,class_weight='auto')

for i in range(1,24):
    test_bool=(subject==i)
    train_bool=(subject!=i)
    Xtest=T[test_bool,:]
    ytest=y[test_bool]
    Xtrain=T[train_bool,:]
    ytrain=y[train_bool]
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
#Save weights and std of SVM coef
std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results",'resolution_3mm',"svm","coef_mean.nii.gz"))

#############################################################################
scores=np.zeros((3,100))

 # SVM & Leave one subject-out with univariate feature selection
for f in range(1,100):
    n=0
    list_predict=list()
    list_true=list()
    coef=np.zeros((24,sum(mask_bool)))
    clf = svm.LinearSVC(C=10e-7,fit_intercept=False,class_weight='auto')

    for i in range(1,24):
        test_bool=(subject==i)
        train_bool=(subject!=i)
        Xtest=T[test_bool,:]
        ytest=y[test_bool]
        Xtrain=T[train_bool,:]
        ytrain=y[train_bool]
        list_true.append(ytest.ravel())
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest=scaler.transform(Xtest)
        selector = SelectPercentile(f_classif, percentile=f)
        clf.fit(selector.fit_transform(Xtrain,ytrain.ravel()), ytrain.ravel())
        #beta[n,:]=clf.coef_

        Xtest=selector.transform(Xtest)
        pred=(clf.predict(Xtest))
        list_predict.append(pred)
        print n 
        n=n+1 
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    recall_scores = recall_score(t,p,pos_label=None, average=None,labels=[0,1])
    scores[0,f]=metrics.accuracy_score(t,p)
    scores[1,f]=recall_scores[0]
    scores[2,f]=recall_scores[1]     

plt.plot(scores[0,:],label="Accuracy")
plt.plot(scores[1,:],label="Specificity")
plt.plot(scores[2,:],label="Sensitivity")
plt.xlabel('Percentiles')
plt.ylabel('Scores')
plt.legend(loc='lower right')

#############################################################################
#Save weights and std of SVM coef
std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"results",'resolution_3mm',"svm","coef_mean.nii.gz"))
#############################################################################

#Display scores of classification (accuracy, precision and recall)

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
#Test with permutations the significance of a classification score
# The p-value is then given by the percentage of runs for which the score
#obtained is greater than the classification score obtained in the first place

nperms=1000
scores_perm = list()
recall_perm = list()

#true_acc= acc

for n in xrange(nperms):
   
    y=np.load(os.path.join(BASE_PATH,'results','svm','resolution_3mm','y.npy'))
    y = np.random.permutation(y)
    list_predict=list()
    list_true=list()
    clf = svm.LinearSVC(C=10e-8,fit_intercept=False,class_weight='auto')

    for i in range(1,24):
        test_bool=(subject==i)
        train_bool=(subject!=i)
        Xtest=T[test_bool,:]
        ytest=y[test_bool]
        Xtrain=T[train_bool,:]
        ytrain=y[train_bool]
        list_true.append(ytest.ravel())
        scaler = preprocessing.StandardScaler().fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest=scaler.transform(Xtest)
        clf.fit(Xtrain, ytrain.ravel())
        pred=(clf.predict(Xtest))
        list_predict.append(pred)
   
     
    t=np.concatenate(list_true)
    p=np.concatenate(list_predict)
    acc=metrics.accuracy_score(t,p)
    scores_perm.append(acc)
    recall_perm.append(recall_score(t,p,pos_label=None, average=None,labels=[0,1]))
    print n


scores_perm=np.array(scores_perm)
pval=np.sum(scores_perm >=0.61)/float(nperms)

recall_perm=np.array(recall_perm)
spe=recall_perm[:,0]
sen=recall_perm[:,1]
pval=np.sum(spe >=0.66)/float(nperms)
pval=np.sum(sen >=0.57)/float(nperms)


plt.hist(scores_perm, 10, label='Permutation scores')
plt.plot(2 * [0.61],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Accuracy')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.68],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Specificity')

plt.hist(sen, 10, label='Permutation scores')
plt.plot(2 * [0.54],plt.ylim(),'--g', linewidth=3)
plt.xlabel('Sensitivity')


#############################################################################
#plot scores of classification per subejct
import matplotlib.pyplot as plt
ind = np.arange(24)

pre=np.zeros((23,1))
rec=np.zeros((23,1))    
acc=np.zeros((23,1))
for i in range (0,23):
    recall_scores = recall_score(list_true[i], list_predict[i],pos_label=None, average=None,labels=[0,1])
    acc[i]=metrics.accuracy_score(list_true[i],list_predict[i])
    pre[i,:]=recall_scores[0]
    rec[i,:]=recall_scores[1]
p, r, f, s = metrics.precision_recall_fscore_support(list_true[i], list_predict[i],pos_label=None, average=None,labels=[0,1])


#Plot accuracy
fig, ax = plt.subplots()
index = np.arange(23)
bar_width = 0.6
opacity = 0.6
plt.bar(index,acc, bar_width,
                 alpha=opacity,
                 color='r')
                 
plt.xlabel('Subjects')
plt.ylabel('Accuracy')
plt.title('Accuracy Score per subject')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'))
plt.legend()   

#Plot recall scores
fig, ax = plt.subplots()
index = np.arange(23)
bar_width = 0.5
opacity = 0.4
r1=plt.bar(index,pre, bar_width,
                 alpha=opacity,
                 color='r',label='Specificity')
r2=plt.bar(index+bar_width,rec, bar_width,
                 alpha=opacity,
                 color='b',label='Sensitivity')                 
 #plot sensitivity             
plt.xlabel('Subjects')
plt.ylabel('Recall scores')
plt.title('SVM specificity and sensitivity by subject')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'))
plt.legend()                 
              
              