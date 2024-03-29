# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:24:03 2016

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
from collections import OrderedDict

BASE_PATH =  '/neurospin/brainomics/2016_AUSZ/results/VBM'
WD = '/neurospin/brainomics/2016_AUSZ/results/VBM/enettv/model_selection'
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
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
    
    penalty_start = 3
    
    alpha = float(key[0])
    l1, l2, tv = alpha * float(key[1]), alpha * float(key[2]), alpha * float(key[3])
    print("l1:%f, l2:%f, tv:%f" % (l1, l2, tv))

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



def scores(key, paths, config, ret_y=False):
    import mapreduce
    print(key)
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values]
    recall_mean_std = np.std([np.mean(precision_recall_fscore_support(
        item["y_true"].ravel(), item["y_pred"])[1]) for item in values]) / np.sqrt(len(values))
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    #prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    #prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    #auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    n_ite = None
    betas = np.hstack([item["beta"] for item in values]).T
    ## Compute beta similarity measures
    # Correlation
    R = np.corrcoef(betas)
    #print R
    R = R[np.triu_indices_from(R, 1)]
    #print R
    # Fisher z-transformation / average
    z_bar = np.mean(1. / 2. * np.log((1 + R) / (1 - R)))
    # bracktransform
    r_bar = (np.exp(2 * z_bar) - 1) /  (np.exp(2 * z_bar) + 1)
    
    # threshold betas to compute fleiss_kappa and DICE
    import array_utils
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in range(betas.shape[0])])
        #print "--", np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1))
#        print np.allclose(np.sqrt(np.sum(betas_t ** 2, 1)) / np.sqrt(np.sum(betas ** 2, 1)), [0.99]*5,
#                           rtol=0, atol=1e-02)
#
#    # Compute fleiss kappa statistics
#    beta_signed = np.sign(betas_t)
#       
#    # Paire-wise Dice coeficient
#    ij = [[i, j] for i in xrange(5) for j in xrange(i+1, 5)]
#    dices = list()
#    for idx in ij:
#        A, B = beta_signed[idx[0], :], beta_signed[idx[1], :]
#        dices.append(float(np.sum((A == B)[(A != 0) & (B != 0)])) / (np.sum(A != 0) + np.sum(B != 0)))
#    dice_bar = np.mean(dices)
#
#    dice_bar = 0.
    
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
    scores['recall_0'] = r[0]
    scores['recall_1'] = r[1]
    scores['recall_mean'] = r.mean()
    scores['recall_mean_std'] = recall_mean_std
    scores['precision_0'] = p[0]
    scores['precision_1'] = p[1]
    scores['precision_mean'] = p.mean()

    scores['support_0'] = s[0]
    scores['support_1'] = s[1]
#    scores['corr']= corr
    scores['beta_r'] = str(R)
    scores['beta_r_bar'] = r_bar
    #scores['beta_fleiss_kappa'] = fleiss_kappa_stat
    #scores['beta_dice'] = str(dices)
    #scores['beta_dice_bar'] = dice_bar
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas_t.shape))
    scores['n_ite'] = n_ite
    scores['param_key'] = key
    if ret_y:
        scores["y_true"], scores["y_pred"] = y_true, y_pred
    return scores
    
    
def reducer(key, values):
    import os, glob, pandas as pd
    os.chdir(os.path.dirname(config_filename()))
    config = json.load(open(config_filename()))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if not p.count("0.8_-1")]

    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="param_key", score="recall_mean"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    print('## Refit scores')
    print('## ------------')
    byparams = groupby_paths([p for p in paths if p.count("refit")], 3) 

    byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}


    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)
    
    print('## doublecv scores by outer-cv and by params')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")], 1)
    for fold, paths_fold in bycv.items():
        print(fold)
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}
        data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
    scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)

#    # rm small l1 with large tv & large l1 with small tv
    rm = (scores_dcv_byparams.prop_non_zeros_mean > 0.5)
        
  
    np.sum(rm)
    scores_dcv_byparams = scores_dcv_byparams[np.logical_not(rm)]

    # model selection on nested cv for 8 cases
#    l2 = scores_dcv_byparams[(scores_dcv_byparams.l1 == 0) & (scores_dcv_byparams.tv == 0)]
#    l2tv = scores_dcv_byparams[(scores_dcv_byparams.l1 == 0) & (scores_dcv_byparams.tv != 0)]
#    l1l2 = scores_dcv_byparams[(scores_dcv_byparams.l1 != 0) & (scores_dcv_byparams.tv == 0)]
    l1l2tv = scores_dcv_byparams[(scores_dcv_byparams.l1 != 0) & (scores_dcv_byparams.tv != 0)]
    # large ans small l1
#    l1l2_ll1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.9) & (scores_dcv_byparams.tv == 0)]
#    l1l2tv_ll1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.9) & (scores_dcv_byparams.tv != 0)]
#    l1l2_sl1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.1) & (scores_dcv_byparams.tv == 0)]
#    l1l2tv_sl1 = scores_dcv_byparams[close(scores_dcv_byparams.l1_ratio, 0.1) & (scores_dcv_byparams.tv != 0)]

    print('## Model selection')
    print('## ---------------')
#    l2 = argmaxscore_bygroup(l2); l2["method"] = "l2"
#    l2tv = argmaxscore_bygroup(l2tv); l2tv["method"] = "l2tv"
#    l1l2 = argmaxscore_bygroup(l1l2); l1l2["method"] = "l1l2"
    l1l2tv = argmaxscore_bygroup(l1l2tv); l1l2tv["method"] = "l1l2tv"

#    l1l2_ll1 = argmaxscore_bygroup(l1l2_ll1); l1l2_ll1["method"] = "l1l2_ll1"
#    l1l2tv_ll1 = argmaxscore_bygroup(l1l2tv_ll1); l1l2tv_ll1["method"] = "l1l2tv_ll1"
#    l1l2_sl1 = argmaxscore_bygroup(l1l2_sl1); l1l2_sl1["method"] = "l1l2_sl1"
#    l1l2tv_sl1 = argmaxscore_bygroup(l1l2tv_sl1); l1l2tv_sl1["method"] = "l1l2tv_sl1"

     #scores_argmax_byfold = pd.concat([l2, l2tv, l1l2, l1l2tv, l1l2_ll1, l1l2tv_ll1, l1l2_sl1, l1l2tv_sl1])
    scores_argmax_byfold = l1l2tv

    print('## Apply best model on refited')
    print('## ---------------------------')
    #scores_l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2.iterrows()], config)
    #scores_l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l2tv.iterrows()], config)
    #scores_l1l2 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2.iterrows()], config)
    scores_l1l2tv = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv.iterrows()], config)

    #scores_l1l2_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_ll1.iterrows()], config)
    #scores_l1l2tv_ll1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_ll1.iterrows()], config)

    #scores_l1l2_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2_sl1.iterrows()], config)
    #scores_l1l2tv_sl1 = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in l1l2tv_sl1.iterrows()], config)

#    scores_cv = pd.DataFrame([["l2"] + scores_l2.values(),
#                  ["l2tv"] + scores_l2tv.values(),
#                  ["l1l2"] + scores_l1l2.values(),
#                  ["l1l2tv"] + scores_l1l2tv.values(),
#
#                  ["l1l2_ll1"] + scores_l1l2_ll1.values(),
#                  ["l1l2tv_ll1"] + scores_l1l2tv_ll1.values(),
#
#                  ["l1l2_sl1"] + scores_l1l2_sl1.values(),
#                  ["l1l2tv_sl1"] + scores_l1l2tv_sl1.values()], columns=["method"] + scores_l2.keys())
    
    
    scores_cv = pd.DataFrame([
                  ["l1l2tv"] + list(scores_l1l2tv.values())], columns=["method"] + list(scores_l1l2tv.keys()))
    print(list(scores_l1l2tv.values()))           
    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_refit', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name='scores_cv', index=False)

##############################################################################


if __name__ == "__main__":
    
    INPUT_DATA_X = '/neurospin/brainomics/2016_AUSZ/results/VBM/X.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_AUSZ/results/VBM/y.npy'
    INPUT_MASK_PATH = '/neurospin/brainomics/2016_AUSZ/results/VBM/mask.nii'
    INPUT_CSV = '/neurospin/brainomics/2016_AUSZ/results/VBM/population.csv'

    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]
    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)

    
    #Outer loop
    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]       
  
             
     
    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        cv["cv%02d/refit" % cv_outer_i] = [tr_val, te]
        cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
        for cv_inner_i, (tr, val) in enumerate(cv_inner):
            cv["cv%02d/cvnested%02d" % (cv_outer_i, cv_inner_i)] = [tr_val[tr], tr_val[val]]
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]
        
    print(list(cv.keys()))  

    # Some QC
    N = float(len(y)); p0 = np.sum(y==0) / N; p1 = np.sum(y==1) / N;
    for k in cv:
        tr, val = cv[k]
        tr, val = np.array(tr), np.array(val)
        print(k, "\t: tr+val=", len(tr) + len(val))
        assert not set(tr).intersection(val)
        #assert abs(np.sum(y[tr]==0)/float(len(y[tr])) - p0) < 0.01
        #assert abs(np.sum(y[tr]==1)/float(len(y[tr])) - p1) < 0.01
        if k.count("refit"):
            te = val
            assert len(tr) + len(te) == len(y)
           # assert abs(len(y[tr])/N - (1 - 1./NFOLDS_OUTER)) < 0.01
        else:
            te = np.array(cv[k.split("/")[0] + "/refit"])[1]
            #assert abs(len(y[tr])/N - (1 - 1./NFOLDS_OUTER) * (1 - 1./NFOLDS_INNER)) < 0.01
            assert not set(tr).intersection(te)
            assert not set(val).intersection(te)
            len(tr) + len(val) + len(te) == len(y)   
            
    # Reduced Parameters grid   
    tv_range = np.hstack([np.arange(0.2, 1, .2)])
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],[.9, .1, 1], [.1, .9, 1]])
    alphas = [.1]

    l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
          
    params = [params.tolist() for params in alphal1l2tv]

#    tv_ratios = [0., .2, .4, .6, .8]
#    l1_ratios = [np.array([1., .1, .9, 1]), np.array([1., .9, .1, 1])]  # [alpha, l1 l2 tv]
#    alphas_l1l2tv = [.01, .1]
#    alphas_l2tv = [round(alpha, 10) for alpha in 10. ** np.arange(-2, 4)]
#    l1l2tv =[np.array([alpha, float(1-tv), float(1-tv), tv]) * l1_ratio
#        for alpha in alphas_l1l2tv for tv in tv_ratios for l1_ratio in l1_ratios]
#    # specific case for without l1 since it supports larger penalties
#    l2tv =[np.array([alpha, 0., float(1-tv), tv])
#        for alpha in alphas_l2tv for tv in tv_ratios]
#    params = l1l2tv + l2tv
#    params = [param.tolist() for param in params]
#    params = {"_".join([str(p) for p in param]):param for param in params}
#    assert len(params) == 50
    
    
    user_func_filename = '/home/ad247405/git/scripts/2016_AUSZ/enettv_doubleCV.py'
    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="model_selectionCV", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="model_selectionCV.csv")
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))