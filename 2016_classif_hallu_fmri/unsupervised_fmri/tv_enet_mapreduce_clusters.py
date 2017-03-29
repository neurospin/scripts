# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 16:23:28 2016

@author: ad247405
"""



import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper
import brainomics.image_atlas
import parsimony.algorithms as algorithms
import parsimony.datasets as datasets
import parsimony.functions.nesterov.tv as nesterov_tv
import parsimony.estimators as estimators
import parsimony.algorithms as algorithms
import parsimony.utils as utils
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score

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



def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["structure"])
    A = tv_helper.A_from_mask(mask_bool)
    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    if resample is not None:
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    else:  # resample is None train == test
        GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k] for idx in [0, 1]]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    import mapreduce as GLOBAL 
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape
    STRUCTURE = GLOBAL.STRUCTURE
    
    global_pen, l1_ratio, tv_ratio = key
    ltv = global_pen * tv_ratio
    ll1 = l1_ratio * global_pen * (1 - tv_ratio)
    ll2 = (1 - l1_ratio) * global_pen * (1 - tv_ratio)

    class_weight="auto" # unbiased
    
    
   
    mask = np.ones(Xtr.shape[0], dtype=bool)
   
    T_IMA= np.load( '/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster1/mapreduce/T_IMA.npy')
    y_IMA= np.load('/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_3rdcomp/cluster1/mapreduce/y_IMA.npy')
    
    T= GLOBAL.DATA["X"]
    y= GLOBAL.DATA["y"]
     
    Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
    T_IMA_diff=T_IMA-Tdiff  
    Xtr=np.vstack((T_IMA_diff, Xtr))
    ytr=np.hstack((y_IMA,ytr))    

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)    
    A = GLOBAL.A
    
    conesta = algorithms.proximal.CONESTA(max_iter=500)
    mod= estimators.LogisticRegressionL1L2TV(ll1,ll2,ltv, A, algorithm=conesta,class_weight=class_weight)
    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def reducer(key, values):
    values = [values[item].load() for item in values]
#
#    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
#            item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
#            / np.sqrt(len(values))
#    recall = [precision_recall_fscore_support(item["y_true"].ravel(),
#                                              item["y_pred"].ravel(),
#                                              average=None)[1] \
#                                              for item in values]
#    support = [precision_recall_fscore_support(item["y_true"].ravel(),
#                                              item["y_pred"].ravel(),
#                                              average=None)[3] \
#                                              for item in values]
#    accuracy_std = np.std([((recall[i][0] * support[i][0] + \
#             recall[i][1] * support[i][1]) \
#                     / (float(support[i][0] + support[i][1]))) \
#                     for i in xrange(len(values))]) \
#             / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc = roc_auc_score(y_true,y_pred)
    print auc
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    betas = np.hstack([item["beta"]  for item in values]).T
    R = np.corrcoef(betas)
    beta_cor_mean = np.mean(R[np.triu_indices_from(R, 1)])
    success = r * s
    success = success.astype('int')
    accuracy = (r[0] * s[0] + r[1] * s[1])
    accuracy = accuracy.astype('int')
    pvalue_accuracy = binom_test(accuracy, s[0] + s[1], p=0.5)
    scores = OrderedDict()
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['accuracy'] = accuracy / float(s[0] + s[1])
    scores['pvalue_accuracy'] = pvalue_accuracy
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['f1_0'] = f[0]
    scores['f1_1'] = f[1]
    scores['f1_mean'] = f.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['beta_cor_mean'] = beta_cor_mean
    # proportion of non zeros elements in betas matrix over all folds
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas)) / \
                                    float(np.prod(betas.shape))
    print scores['accuracy']                             
    return scores
##############################################################################


if __name__ == "__main__":
    WD = '/neurospin/brainomics/2016_classif_hallu_fmri/unsupervised_fmri/clustering_only_hallu/cluster_randomB'

    INPUT_DATA_X = os.path.join(WD,'T_clusterB.npy')
    INPUT_DATA_y = os.path.join(WD,'y_clusterB.npy')
    INPUT_DATA_subject = os.path.join(WD,'subject_clusterB.npy')
    INPUT_MASK_PATH = os.path.join(WD,"mask.nii.gz")


    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    subject = np.load(INPUT_DATA_subject)
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=23)]
   
    for i in range(1,24):
        test_bool=(subject==i)
        train_bool=(subject!=i)
        cv[i-1][0] = np.asarray(np.where(subject!=i)).tolist()[0]
        cv[i-1][1] = np.asarray(np.where(subject==i)).tolist()[0]
        
        
    params= [[0.1, 0.1, 0.1],[0.1, 0.1, 0.5],[0.1,0.1,1e-06],[0.01, 0.1, 0.1],[0.01, 0.1, 0.5],[0.01,0.1,1e-06],[0.1,0.5,0.5],[0.1,0.5,1e-06],[0.1,0.5,0.1]]
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2016_classif_hallu_fmri", "unsupervised_fmri",
        "tv_enet_mapreduce_clusters.py")
    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="results", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))


