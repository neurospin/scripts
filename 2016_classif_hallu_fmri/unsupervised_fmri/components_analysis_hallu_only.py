# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:51:38 2016

@author: ad247405
"""

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

INPUT_BASE_DIR = "/neurospin/brainomics/2014_pca_struct/fmri/fmri_5folds_hallu_only/fmri_5folds"
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

params=np.array(('struct_pca', '0.1', '0.8', '0.5')) 

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


plt.plot(subject[y==1], projections[:,1],'o')
plt.xlabel("Patients")
plt.ylabel("score on 2nd component")



#Clustering of PreH block
##############################################################################
mod = cluster.KMeans(n_clusters=2)
predict = mod.fit_predict(projections[:,1].reshape(83,1))

#np.random.shuffle(predict)


plt.plot(projections[:,0][predict==0],projections[:,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[:,0][predict==1],projections[:,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 2')
plt.legend()

plt.plot(projections[:,0][predict==0],projections[:,1][predict==0],'o',label = 'cluster A')
plt.plot(projections[:,0][predict==1],projections[:,1][predict==1],'o',label = 'cluster B')
plt.xlabel('component 0')
plt.ylabel('component 1')
plt.legend()

plt.plot(projections[:,1][predict==0],projections[:,2][predict==0],'o',label = 'cluster A')
plt.plot(projections[:,1][predict==1],projections[:,2][predict==1],'o',label = 'cluster B')
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.legend()

data = [projections[:,1][predict==0],projections[:,1][predict==1]]
plt.boxplot(data)
plt.plot(predict+1,projections[:,1],'o', linewidth = 1)
plt.gca().xaxis.set_ticklabels(['Cluster A', 'cluster B'])
plt.ylabel('Score on 2nd component')
##############################################################################
#SVM & Leave one subject-out - no feature selection - WITH IMA samples 
#############################################################################
#predict = np.random.randint(2,size=83)

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
 
 
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_only_hallu/cluster_randomB/T_clusterB.npy',T)
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_only_hallu/cluster_randomB/subject_clusterB.npy',subject)
np.save('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_only_hallu/cluster_randomB/y_clusterB.npy',y)
    
 
 
#SVM & Leave one subject-out - no feature selection - WITH IMA samples
###########################################################################

n=0
list_predict=list()
list_true=list()
coef=np.zeros((23,63966))
#coef=np.zeros((24,8028))
clf = svm.LinearSVC(C=1e-3,fit_intercept=True,class_weight='auto')

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
#acc=metrics.accuracy_score(t,p)
auc = roc_auc_score(t,p)
pre=recall_scores[0]
rec=recall_scores[1]
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
