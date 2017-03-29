# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:11:36 2015

@author: ad247405
"""


import nilearn.signal
import re
import glob
import os
import nibabel as nibabel
import numpy as np
import pandas as pd
import os
import numpy as np
import brainomics.image_atlas
import matplotlib.pyplot as plt
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from sklearn.metrics import precision_recall_fscore_support,recall_score
from sklearn.linear_model import LogisticRegression
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
import mapreduce
from parsimony.utils.penalties import l1_max_logistic_loss
import csv
from multiprocessing import Process, Manager
import traceback

#############################################################################
BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri"
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

T_all=np.vstack((T_IMA_diff,T))
y_all=np.hstack((y_IMA,y))
#
#bdiff=np.mean(b_IMA,axis=0)-np.mean(b[y==0],axis=0)
#b_IMA_diff=b_IMA-bdiff
#T_IMA_diff=b_IMA_diff
#T=b
#############################################################################    
#parameter grid 
#tv_range = np.hstack([np.arange(0, 1., .1), [0.05, 0.01, 0.005, 0.001,]])
#tv_range = np.hstack([np.arange(0, 1., .1)])
#ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1], [.9, .1, 1],[.1, .9, 1], [.01, .99, 1], [.001, .999, 1]])
#l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
#l1l2tv.append(np.array([[0., 0., 1.]]))
#l1l2tv = np.concatenate(l1l2tv)
#params = [params.tolist() for params in l1l2tv]

params= [[0.1,0.5,0.5]]

#############################################################################    

#File to store classification scores
f=open(os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','parameters_scores_april2.csv'),'wb')
c=csv.writer(f,delimiter=',')
c.writerow(["alpha","l1","tv","accuracy","recall_0","recall_1","precision_0","precision_1","auc"])


# Empirically set the global penalty, based on maximum l1 penaly
#alpha = l1_max_logistic_loss(T_all, y_all)


conesta = algorithms.proximal.CONESTA(max_iter=500)
A= nesterov_tv.linear_operator_from_mask(mask_bool)

# Messages for communication between processes
FLAG_STOP_PROCESS     = "STOP_WORK"
FLAG_PROCESS_FINISHED = "PROCESS_HAS_FINISHED"
nb_processes=1
 # Data structures for parallel processing
manager = Manager()  # multiprocessing.Manager()
work_queue, result_queue = manager.Queue(), manager.Queue()

# Add jobs in work_queue
for p in params:
    #print p
    work_queue.put(p)

# Add poison pills to stop the remote workers
# When a process gets this job, it will stop
for n in range(nb_processes):
    work_queue.put(FLAG_STOP_PROCESS)

def parallel_worker(work_queue, result_queue):
    """ Function to make complete_preprocessing work in parallel processing.
    """
    while True:
        new_work = work_queue.get()
        if new_work == FLAG_STOP_PROCESS:
            result_queue.put(FLAG_PROCESS_FINISHED)
            break
        p = new_work
        try:
            r=fitting(p)
            result_queue.put(r)
        
        except Exception as e:
            e.message += "\nUnknown error happened for %s" % str(p)
            e.message += "\n" + traceback.format_exc()
            result_queue.put(e.message)


def fitting(p):
    global_pen,l1_ratio, tv_ratio, = p[0],p[1],p[2]
    
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)
        
    
    #l1, l2, tv= alpha * float(p[0]), alpha * float(p[1]), alpha * float(p[2])
    clf= estimators.LogisticRegressionL1L2TV(ll1,ll2,ltv, A, algorithm=conesta)
    n=0  
    list_predict=list()
    list_true=list()
    list_proba_pred=list()
    coef=np.zeros((23,63966))     
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
        coef[n,:]=clf.beta[:,0]
        pred=(clf.predict(Xtest))
        list_predict.append(pred)
        proba_pred = clf.predict_probability(Xtest)
        list_proba_pred.append(proba_pred) 
        n=n+1
        print n
       

    true=np.concatenate(list_true)
    pred=np.concatenate(list_predict)
    proba_pred=np.concatenate(list_proba_pred)
    precision, recall, f, s = precision_recall_fscore_support(true,pred, average=None)
    acc=metrics.accuracy_score(true,pred)
    auc = roc_auc_score(true,pred)
    current=[global_pen,l1_ratio,tv_ratio,acc,recall[0],recall[1],precision[0],precision[1],auc]
    np.save(os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','betas_subj.npy'),coef)    
    return current


# Define processes
workers = []
for i in range(nb_processes):
    worker = Process(target=parallel_worker, args=(work_queue, result_queue))
    worker.daemon = True
    workers.append(worker)
    worker.start()


# Process results and log everything
nb_finished_processes = 0
try:
    while True:
        new_result = result_queue.get()
        if new_result == FLAG_PROCESS_FINISHED:
            nb_finished_processes += 1
            print("Finished processes: %d/%d" % (nb_finished_processes,
                                                       nb_processes))
            if nb_finished_processes == nb_processes:
                break
        elif type(new_result) is list:
            c.writerow(new_result)
            print new_result
        else:
            print("error " + new_result)
except KeyboardInterrupt:  # To stop if user uses ctrl+c
    print("KeyboardInterrupt: stopping processes.")
    for worker in workers:
        worker.terminate()
        worker.join()

f.close()
#############################################################################    

#############################################################################
#Save Discriminative map
np.save(os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','betas_mean.npy'),coef)
#std_coef=coef.std(axis=0)
mean_coef=coef.mean(axis=0)
arr = np.zeros(mask_bool.shape);
arr[mask_bool] = mean_coef
out_im = nibabel.Nifti1Image(arr, affine=babel_mask.get_affine())
out_im.to_filename(os.path.join(BASE_PATH,"toward_on","Logistic_L1_L2_TV_with_HC","beta_mean.nii.gz"))
#############################################################################


#############################################################################

#plot acc,spe and sen while TV is increased  
data = pd.read_csv((os.path.join(BASE_PATH,'toward_on','Logistic_L1_L2_TV_with_HC','parameters_scores.csv')))  
data = data[(data.tv >= 0.05) | (data.tv == 0.0)]
data["balanced_acc"]=(data.recall_0+data.recall_1)/2

full_tv = data[(data.tv == 1)]

d1=data[np.round((data.l1)/(data.l2),3) == 0.001]
d2=data[np.round((data.l1)/(data.l2),3) == 0.01]
d3=data[np.round((data.l1)/(data.l2),3) == 0.111]
d4=data[np.round((data.l1)/(data.l2),3) == 1]
d5=data[np.round((data.l1)/(data.l2),3) == 9]


d1 = d1.append(full_tv) # add full tv for all lines
d2 = d2.append(full_tv) # add full tv for all lines
d3 = d3.append(full_tv) # add full tv for all lines
d4 = d4.append(full_tv) # add full tv for all lines
d5 = d5.append(full_tv) # add full tv for all lines


d1=d1.sort("tv")
d2=d2.sort("tv")
d3=d3.sort("tv")
d4=d4.sort("tv")
d5=d5.sort("tv")


plt.plot(d4.tv, d1.balanced_acc,"green",label=r'$\lambda_1/\lambda_2 = 0.001 $',linewidth=2)
plt.plot(d2.tv, d2.balanced_acc,"blue",label=r'$\lambda_1/\lambda_2 = 0.01 $',linewidth=2)
plt.plot(d5.tv, d3.balanced_acc,"orange",label=r'$\lambda_1/\lambda_2 = 0.1 $',linewidth=2)
plt.plot(d1.tv, d4.balanced_acc,"red",label=r'$\lambda_1/\lambda_2 = 1 $',linewidth=2)
plt.plot(d3.tv, d5.balanced_acc,"black",label=r'$\lambda_1/\lambda_2 = 10 $',linewidth=2)

#plt.plot(d1.tv, d1.recall_1,"red",label="TVl1l2- (l1=0.1,l2=0.9)",linewidth=2)
#plt.plot(d2.tv, d2.accuracy,"blue",label="TVl1l2(l1=0.9,l2=0.1",linewidth=2)
plt.ylim(.4, .9)
plt.ylabel(" Balanced Accuracy")
#plt.ylabel("Specificity")
#plt.ylabel("Sensitivity")
plt.xlabel(r'TV ratio: $\lambda_{tv}/(\lambda_1 + \lambda_2 + \lambda_{tv})$')
plt.grid(True)
plt.legend()











import csv


error = np.zeros((165))
for i in range(165):
    if true[i]==pred[i]:
        error[i]=0
        
    else:
        error[i]=1  
        
    
a=np.array((subject,true,pred.reshape(165),error,projections[:,2]))
a=a.T
   

df = pd.DataFrame(a,columns=['subject', 'true', 'prediction','error','score_3rd'])
df.to_csv('/neurospin/brainomics/2016_classif_hallu_fmri/toward_on/Logistic_L1_L2_TV_with_HC/0.1_0.1_0.1_classification_file.csv')