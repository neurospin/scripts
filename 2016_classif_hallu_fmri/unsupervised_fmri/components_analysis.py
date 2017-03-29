# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 09:20:50 2016

@author: ad247405
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from brainomics import plot_utilities
import parsimony.utils.check_arrays as check_arrays
from sklearn import cluster
from sklearn import svm
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
import brainomics.image_atlas
import nibabel as nibabel
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import precision_recall_fscore_support,recall_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import roc_auc_score, recall_score

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds/"
INPUT_DIR = os.path.join(INPUT_BASE_DIR,"results")


N_COMP = 3
EXAMPLE_FOLD = 0

INPUT_COMPONENTS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'components.npz')
INPUT_PROJECTIONS_FILE_FORMAT = os.path.join(INPUT_DIR,
                                            '{fold}',
                                            '{key}',
                                            'X_train_transform.npz')

params=np.array(('struct_pca', '0.1', '0.5', '0.5')) 

components  =np.zeros((63966, 3))
fold=0
key = '_'.join([str(param)for param in params])
print "process", key
name=params[0]

components_filename = INPUT_COMPONENTS_FILE_FORMAT.format(fold=fold,key=key)
projections_filename = INPUT_PROJECTIONS_FILE_FORMAT.format(fold=fold,key=key) 

      
components = np.load(components_filename)['arr_0']
projections = np.load(projections_filename)['arr_0']
assert projections.shape[1] == components.shape[1]

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))


# Plot of components according to clinical status
#####################################################################


plt.plot(projections[y==0,0],projections[y==0,1],'o',label = " No hallucinations")
plt.plot(projections[y==1,0],projections[y==1,1],'o',label = " Preceeding hallucinations")
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp0_1.png',format='png')


plt.plot(projections[y==0,0],projections[y==0,2],'o',label = " No hallucinations")
plt.plot(projections[y==1,0],projections[y==1,2],'o',label = " Preceeding hallucinations")
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp0_2.png',format='png')


plt.plot(projections[y==0,1],projections[y==0,2],'o',label = " No hallucinations")
plt.plot(projections[y==1,1],projections[y==1,2],'o',label = " Preceeding hallucinations")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp1_2.png',format='png')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projections[y==0,0],projections[y==0,1],projections[y==0,2],c='b',label = " No hallucinations")
ax.scatter(projections[y==1,0],projections[y==1,1],projections[y==1,2],c='g',label = " Preceeding hallucinations")
ax.set_xlabel('component 0')
ax.set_ylabel('component 1')
ax.set_zlabel('component 2')
plt.legend()
plt.savefig('/neurospin/brainomics/2014_pca_struct/fmri/components_analysis/comp1_2_3.png',format='png')
#####################################################################

#Clustering of PreH block
##############################################################################
mod = cluster.KMeans(n_clusters=2)
predict = mod.fit_predict(projections[y==1,2].reshape(83,1))


plt.plot(projections[y==1,0][predict==0],projections[y==1,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,0][predict==1],projections[y==1,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()

plt.plot(projections[y==1,0][predict==0],projections[y==1,1][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,0][predict==1],projections[y==1,1][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()

plt.plot(projections[y==1,1][predict==0],projections[y==1,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[y==1,1][predict==1],projections[y==1,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()

data = [projections[y==1,2][predict==0],projections[y==1,2][predict==1]]
plt.boxplot(data)
plt.plot(predict+1,projections[y==1,2],'o', linewidth = 1)
plt.gca().xaxis.set_ticklabels(['Cluster A', 'cluster B'])
plt.ylabel('Score on 3rd component')
##############################################################################
#SVM & Leave one subject-out - no feature selection - WITH IMA samples 
#############################################################################


T=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
b=np.load(os.path.join(BASE_PATH,'toward_on','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))

T_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','T_IMA.npy'))
b_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_IMA.npy'))
y_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','y_IMA.npy'))
subject_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','subject_IMA.npy'))
#   
 
Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
T_IMA_diff=T_IMA-Tdiff

 
subject_off = subject[y==0]
subject_on = subject[y==1]
subject_cluster0 = subject[y==1][predict==0]
subject_cluster1 = subject[y==1][predict==1]

T_off = T[y==0]
y_off = np.zeros((T[y==0].shape[0]))

T_cluster0 = T[y==1,:][predict==0]
y_cluster0 = np.ones((T_cluster0.shape[0]))
T_cluster1 = T[y==1,:][predict==1]
y_cluster1 = np.ones((T_cluster1.shape[0]))
 
T=np.vstack((T_off,T_cluster1))
y=np.hstack((y_off,y_cluster1))
subject = np.hstack((subject_off,subject_cluster1))
 
 
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster0/mapreduce/T_cluster1.npy',T)
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster0/mapreduce/subject_cluster1.npy',subject)
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster0/mapreduce/y_cluster1.npy',y)
    
 
 
#SVM & Leave one subject-out - no feature selection - WITH IMA samples
###########################################################################

n=0
list_predict=list()
list_true=list()
coef=np.zeros((23,63966))
#coef=np.zeros((24,8028))
clf = svm.LinearSVC(C=100,fit_intercept=True,class_weight='auto')

for i in range(1,24):
    test_bool=(subject==i)
    train_bool=(subject!=i)
    Xtest=T[test_bool,:]
    ytest=y[test_bool]
    Xtrain=np.vstack((T_IMA_diff,T[train_bool,:]))
    ytrain=np.hstack((y_IMA,y[train_bool]))
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
auc = roc_auc_score(t,p)
pre=recall_scores[0]
rec=recall_scores[1]
print acc
print auc
print pre
print rec




np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/weightmap_cluster0.npy',coef)

np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/weightmap_cluster1.npy',coef)
#############################################################################
#Save weights and std of SVM coef
std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/mean_weightmap_cluster1.nii')

arr = np.zeros(mask_bool.shape);
arr[mask_bool] =beta.reshape(63966)
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster1/weight_map_cluster1_0.1_0.5_0.5.nii')

arr = np.zeros(mask_bool.shape);
arr[mask_bool] =beta.reshape(63966)
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster0/weight_map_cluster0_0.1_0.1_0.1.nii')


###################################################################################
#ElasticNet with TV penalty 
#############################################################################

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"

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


mod = cluster.KMeans(n_clusters=2)
predict = mod.fit_predict(projections[y==1,:])
sum(predict)
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/cluster0/mapreduce/cluster_label.npy',predict)


T=np.load(os.path.join(BASE_PATH,'toward_on','svm','T.npy'))
b=np.load(os.path.join(BASE_PATH,'toward_on','svm','betas.npy'))
y=np.load(os.path.join(BASE_PATH,'toward_on','svm','y.npy'))
subject=np.load(os.path.join(BASE_PATH,'toward_on','svm','subject.npy'))

T_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','T_IMA.npy'))
b_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','betas_IMA.npy'))
y_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','y_IMA.npy'))
subject_IMA=np.load(os.path.join(BASE_PATH,'toward_on','svm_with_HC','subject_IMA.npy'))
#   

Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
T_IMA_diff=T_IMA-Tdiff
 
subject_off = subject[y==0]
subject_on = subject[y==1]
subject_cluster0 = subject[y==1][predict==0]
subject_cluster1 = subject[y==1][predict==1]

T_off = T[y==0]
y_off = np.zeros((T[y==0].shape[0]))

T_cluster0 = T[y==1,:][predict==0]
y_cluster0 = np.ones((T_cluster0.shape[0]))
T_cluster1 = T[y==1,:][predict==1]
y_cluster1 = np.ones((T_cluster1.shape[0]))
 
T=np.vstack((T_off,T_cluster0))
y=np.hstack((y_off,y_cluster0))
subject = np.hstack((subject_off,subject_cluster1))
 
