# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:00:01 2016

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

BASE_PATH = '/neurospin/brainomics/2016_deptms'
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = np.load(config["structure"])
    A = tv_helper.A_from_mask(STRUCTURE)
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

    penalty_start = 2
        
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
    mod= estimators.LogisticRegressionL1L2TV(l1,l2,tv, A, algorithm=conesta,class_weight=class_weight,penalty_start=penalty_start)
    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.predict_probability(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, proba_pred=proba_pred, beta=mod.beta,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


def reducer(key, values):
    print key
    values = [values[item].load() for item in values]
    #FOLD O is a refit on all samples. 
    values = values[1:]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = [item["proba_pred"].ravel() for item in values]
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    betas = np.hstack([item["beta"] for item in values]).T    
    # threshold betas to compute fleiss_kappa and DICE
    import array_utils
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in xrange(betas.shape[0])])
    success = r * s
    success = success.astype('int')
    accuracy = (r[0] * s[0] + r[1] * s[1])
    accuracy = accuracy.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pvalue_recall0 = binom_test(success[0], s[0], 1 - prob_class1)
    pvalue_recall1 = binom_test(success[1], s[1], prob_class1)
    pvalue_accuracy = binom_test(accuracy, s[0] + s[1], p=0.5)
    scores = OrderedDict()
    try:    
        a, l1, l2 , tv  = [float(par) for par in key.split("_")]
        scores['a'] = a
        scores['l1'] = l1
        scores['l2'] = l2
        scores['tv'] = tv
        left = float(1 - tv)
        if left == 0: left = 1.
        scores['l1_ratio'] = float(l1) / left
    except:
        pass
    scores['recall_mean'] = r.mean()
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['accuracy'] = accuracy/ float(len(y_true))
    scores['pvalue_recall_0'] = pvalue_recall0
    scores['pvalue_recall_1'] = pvalue_recall1
    scores['pvalue_accuracy'] = pvalue_accuracy
    scores['max_pvalue_recall'] = np.maximum(pvalue_recall0, pvalue_recall1)
    scores['recall_mean'] = r.mean()
    scores['auc'] = auc
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas_t.shape))
    #scores['param_key'] = key
    return scores
##############################################################################


if __name__ == "__main__":
    BASE_PATH = '/neurospin/brainomics/2016_deptms'
    WD = '/neurospin/brainomics/2016_deptms/Freesurfer/results/enettv/no_model_selection5folds'

    INPUT_DATA_X = '/neurospin/brainomics/2016_deptms/Freesurfer/data/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_deptms/Freesurfer/data/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_deptms/Freesurfer/data/mask.npy'
    INPUT_CSV = '/neurospin/brainomics/2016_deptms/population.csv'

    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]
    N_FOLDS = 5

    #############################################################################
    ## Create config file
    y  = np.load(INPUT_DATA_y).ravel()
    skf = StratifiedKFold(n_folds=N_FOLDS,y=y)
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=N_FOLDS)]
    if cv[0] is not None: # Make sure first fold is None
        cv.insert(0, None)   
        null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
        cv[0] = np.asarray(null_resampling).tolist()
   
     # Reduced Parameters grid   
    tv_range = tv_ratios = [.2, .4, .6, .8]
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],[.9, .1, 1], [.1, .9, 1],[.3,.7,1],[.7,.3,1]])
    alphas = [0.01,.1,0.5]

    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
          
    params = [params.tolist() for params in alphal1l2tv]

    user_func_filename = '/home/ad247405/git/scripts/2016_deptms/03_enettv_no_model_selection.py'
    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="results", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))