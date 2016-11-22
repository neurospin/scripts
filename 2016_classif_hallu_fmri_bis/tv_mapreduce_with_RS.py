# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:55:06 2016

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
import pandas as pd

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.DATA_RS = GLOBAL.load_data(config["data_RS"])
    STRUCTURE = nibabel.load(config["structure"])
    A = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}

def mapper(key, output_collector):
    import mapreduce as GLOBAL 
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]

    T_RS = GLOBAL.DATA_RS["X_RS"] 
    y_RS = GLOBAL.DATA_RS["y_RS"] 
    T = GLOBAL.DATA["X"]
    y = GLOBAL.DATA["y"]
    Tdiff=np.mean(T_RS,axis=0)-np.mean(T[y==0],axis=0)
    T_RS_diff=T_RS-Tdiff 
    Xtr=np.vstack((T_RS_diff,Xtr))
    ytr=np.hstack((y_RS,ytr))
    
    
    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print "l1:%f, l2:%f, tv:%f" % (l1, l2, tv)

    class_weight="auto" # unbiased
    
    mask = np.ones(Xtr.shape[0], dtype=bool)
   
    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)    
    A = GLOBAL.A
    
    conesta = algorithms.proximal.CONESTA(max_iter=500)
    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,class_weight=class_weight)
    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret

def reducer(key, values):
    values = [values[item].load() for item in values]
    #FOLD O is a refit on all samples. DOn't take into account in test.
    values = values[1:]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
            item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) \
            / np.sqrt(len(values))
    recall = [precision_recall_fscore_support(item["y_true"].ravel(),
                                              item["y_pred"].ravel(),
                                              average=None)[1] \
                                              for item in values]
    support = [precision_recall_fscore_support(item["y_true"].ravel(),
                                              item["y_pred"].ravel(),
                                              average=None)[3] \
                                              for item in values]
    accuracy_std = np.std([((recall[i][0] * support[i][0] + \
             recall[i][1] * support[i][1]) \
                     / (float(support[i][0] + support[i][1]))) \
                     for i in xrange(len(values))]) \
             / np.sqrt(len(values))
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
    import array_utils
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in xrange(betas.shape[0])])
    scores = OrderedDict()
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['accuracy'] = accuracy / float(s[0] + s[1])
    scores['pvalue_accuracy'] = pvalue_accuracy
    scores['accuracy_std'] = accuracy_std
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()
    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
    scores['beta_cor_mean'] = beta_cor_mean
    # proportion of non zeros elements in betas matrix over all folds
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas_t.shape))
    print scores['accuracy']                             
    return scores
##############################################################################


if __name__ == "__main__":
    BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
    WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/enettv/with_RS_no_model_selection'
    INPUT_DATA_X = os.path.join(BASE_PATH,'results_nov/multivariate_analysis/data','T.npy')
    INPUT_DATA_y = os.path.join(BASE_PATH,'results_nov/multivariate_analysis/data','y_state.npy')
    INPUT_DATA_subject = os.path.join(BASE_PATH,'results_nov/multivariate_analysis/data','subject.npy')   
    INPUT_DATA_RS_X = os.path.join(BASE_PATH,'results_nov/multivariate_analysis/data','T_RS.npy')
    INPUT_DATA_RS_y = os.path.join(BASE_PATH,'results_nov/multivariate_analysis/data','y_RS.npy')
    INPUT_MASK_PATH = os.path.join(BASE_PATH,'results_nov',"multivariate_analysis","data","MNI152_T1_3mm_brain_mask.nii.gz")
    INPUT_CSV = os.path.join(BASE_PATH,"population26oct.txt")

    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]


    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    subject = np.load(INPUT_DATA_subject)
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=number_subjects+1)]
    cv[0][0] = np.asarray(np.where(subject!=340)).tolist()[0]
    cv[0][1] = np.asarray(np.where(subject!=340)).tolist()[0]
    
    for i in range(1,number_subjects+1):
        test_bool=(subject==(i-1))
        train_bool=(subject!=(i-1))
        cv[i][0] = np.asarray(np.where(subject!=(i-1))).tolist()[0]
        cv[i][1] = np.asarray(np.where(subject==(i-1))).tolist()[0]

    
     # Reduced Parameters grid   
    tv_range = tv_ratios = [.2, .4, .6, .8]
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],[.9, .1, 1], [.1, .9, 1],[.3,.7,1],[.7,.3,1]])
    alphas = [0.01,.1,0.5]

    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
          
    params = [params.tolist() for params in alphal1l2tv]


    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2016_classif_hallu_fmri_bis", "tv_mapreduce_with_RS.py")
    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  data_RS=dict(X_RS=INPUT_DATA_RS_X, y_RS=INPUT_DATA_RS_y),
                  subject = INPUT_DATA_subject,
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="results", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))