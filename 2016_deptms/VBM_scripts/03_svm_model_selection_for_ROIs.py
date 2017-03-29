#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 18:21:24 2016

@author: ad247405
"""


import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing,metrics
from sklearn.metrics import roc_auc_score
import pandas as pd
import shutil
from sklearn import svm

BASE_PATH= '/neurospin/brainomics/2016_deptms'
WD = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/svm_ROIs/Roiho-frontalPole"
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV_5folds.xlsx")
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    

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

    
    c = float(key[0])
    print("c:%f" % (c))

    class_weight="auto" # unbiased
    
    mask = np.ones(Xtr.shape[0], dtype=bool)
   
    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xte=scaler.transform(Xte)    
    
    mod = svm.LinearSVC(C=c,fit_intercept=False,class_weight= class_weight)

    mod.fit(Xtr, ytr.ravel())
    y_pred = mod.predict(Xte)
    proba_pred = mod.decision_function(Xte)
    ret = dict(y_pred=y_pred, y_true=yte,proba_pred = proba_pred, beta=mod.coef_,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret



def scores(key, paths, config):
    import mapreduce
    print(key)
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    prob_pred = [item["proba_pred"].ravel() for item in values]
    prob_pred = np.concatenate(prob_pred)
    p, r, f, s = metrics.precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, prob_pred) #area under curve score.
    betas = np.hstack([item["beta"] for item in values]).T    
    # threshold betas to compute fleiss_kappa and DICE
    import array_utils
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in range(betas.shape[0])])
    #Compute pvalue                  
    success = r * s
    success = success.astype('int')
    prob_class1 = np.count_nonzero(y_true) / float(len(y_true))
    pvalue_recall0_true_prob = binom_test(success[0], s[0], 1 - prob_class1,alternative = 'greater')
    pvalue_recall1_true_prob = binom_test(success[1], s[1], prob_class1,alternative = 'greater')
    pvalue_recall0_unknwon_prob = binom_test(success[0], s[0], 0.5,alternative = 'greater')
    pvalue_recall1_unknown_prob = binom_test(success[1], s[1], 0.5,alternative = 'greater')
    pvalue_recall_mean = binom_test(success[0]+success[1], s[0] + s[1], p=0.5,alternative = 'greater')
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
    scores["auc"] = auc
    scores['pvalue_recall0_true_prob_one_sided'] = pvalue_recall0_true_prob
    scores['pvalue_recall1_true_prob_one_sided'] = pvalue_recall1_true_prob
    scores['pvalue_recall0_unknwon_prob_one_sided'] = pvalue_recall0_unknwon_prob
    scores['pvalue_recall1_unknown_prob_one_sided'] = pvalue_recall1_unknown_prob
    scores['pvalue_recall_mean'] = pvalue_recall_mean
    scores['prop_non_zeros_mean'] = float(np.count_nonzero(betas_t)) / \
                                    float(np.prod(betas.shape))
    scores['param_key'] = key
    return scores
    
    
def reducer(key, values):
    import os, glob, pandas as pd
    os.chdir(os.path.dirname(config_filename()))
    config = json.load(open(config_filename()))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if p.count("0.1")]

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
    byparams = groupby_paths([p for p in paths if p.count("all") and not p.count("all/all")],3) 
    byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)
    
    print('## doublecv scores by outer-cv and by params')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested")],1)
    for fold, paths_fold in list(bycv.items()):
        print(fold)
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}
        data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
        scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)


    print('## Model selection')
    print('## ---------------')
    svm = argmaxscore_bygroup(scores_dcv_byparams); svm["method"] = "svm"
    
    scores_argmax_byfold = svm

    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_svm = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "all", row["param_key"]) for index, row in svm.iterrows()], config)

   
    scores_cv = pd.DataFrame([["svm"] + list(scores_svm.values())], columns=["method"] + list(scores_svm.keys()))
   
         
    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_all', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name='scores_cv', index=False)

##############################################################################

if __name__ == "__main__":
    BASE_PATH = '/neurospin/brainomics/2016_deptms'
    INPUT_ROIS_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/ROI_labels.csv"
    INPUT_CSV = "/neurospin/brainomics/2016_deptms/analysis/VBM/population.csv"
    INPUT_DATA_y = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/y.npy"
    INPUT_ROIS_DATA = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/ROIs_data"
    MASK_PATH = "/neurospin/brainomics/2016_deptms/analysis/VBM/data/mask.nii"
    OUTPUT_ENETTV = "/neurospin/brainomics/2016_deptms/analysis/VBM/results/svm_ROIs"

    
    penalty_start = 3
    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]
    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5

    
    #########################################################################
    ## Read ROIs csv
    atlas = []
    dict_rois = {}
    df_rois = pd.read_csv(INPUT_ROIS_CSV)
    for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
        cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
        label_ho = cur["label_ho"].values[0]
        atlas_ho = cur["atlas_ho"].values[0]
        roi_name = cur["ROI_name_deptms"].values[0]
        if ((not cur.isnull()["atlas_ho"].values[0])
            and (not cur.isnull()["ROI_name_deptms"].values[0])):
            if ((not roi_name in dict_rois)
              and (roi_name != "Maskdep-sub")
              and (roi_name != "Maskdep-cort")):
                labels = np.asarray(label_ho.split(), dtype="int")
                dict_rois[roi_name] = [labels]
                dict_rois[roi_name].append(atlas_ho)

    rois = list(set(df_rois["ROI_name_deptms"].values.tolist()))
    rois = [x for x in rois if str(x) != 'nan']
    rois.remove('Maskdep')
   
    #########################################################################
     ## Build config file for all roi
    for roi in rois:
        print ("ROI", roi)
        WD = os.path.join(OUTPUT_ENETTV,roi)
        if not os.path.exists(WD):
            os.makedirs(WD)
        INPUT_MASK = os.path.join(INPUT_ROIS_DATA,'mask_' + roi + '.nii')
        # copy X, y, mask file names in the current directory

        INPUT_DATA_X = os.path.join(INPUT_ROIS_DATA,
                                    'X_'+roi + '.npy')
        shutil.copy2(INPUT_DATA_X, os.path.join(WD, 'X.npy'))
        
        shutil.copy2(INPUT_DATA_y, os.path.join(WD, 'y.npy'))
        
        INPUT_MASK = os.path.join(INPUT_ROIS_DATA,
                                     'mask_' + roi + '.nii')
        shutil.copy2(INPUT_MASK, os.path.join(WD, 'mask.nii'))  


    
    #############################################################################
    ## Create config file
        y = np.load(INPUT_DATA_y)
    
        cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
        if cv_outer[0] is not None: # Make sure first fold is None
            cv_outer.insert(0, None)   
            null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
            cv_outer[0] = null_resampling
                
    #     
        import collections
        cv = collections.OrderedDict()
        for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
            if cv_outer_i == 0:
                cv["all/all"] = [tr_val, te]
            else:    
                cv["cv%02d/all" % (cv_outer_i -1)] = [tr_val, te]
                cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
                for cv_inner_i, (tr, val) in enumerate(cv_inner):
                    cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
        for k in cv:
            cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]
    
           
        print((list(cv.keys())))  
    
    #    # Full Parameters grid   
    #    tv_range = tv_ratios = [.2, .4, .6, .8]
    #    ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],[.9, .1, 1], [.1, .9, 1],[.3,.7,1],[.7,.3,1]])
    #    alphas = [0.01,.1,0.5]
    
         # Reduced Parameters grid   
        tv_range = tv_ratios = [.2, .4, .6, .8]
        ratios = np.array([[1., 0., 1], [0., 1., 1], [.5, .5, 1],[.9, .1, 1], [.1, .9, 1]])
        alphas = [.1]
    
        l1l2tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
        l1l2tv = np.concatenate(l1l2tv)
        alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
              
        params = [params.tolist() for params in alphal1l2tv]
    
        
        user_func_filename = "/home/ad247405/git/scripts/2016_deptms/VBM_scripts/03_svm_model_selection_for_ROIs.py"
        
        config = dict(data=dict(X=os.path.join(WD, 'X.npy'), y=os.path.join(WD, 'y.npy')),
                      params=params, resample=cv,
                      structure=os.path.join(WD, 'mask.nii'),
                      map_output="model_selectionCV", 
                      user_func=user_func_filename,
                      reduce_input="results/*/*",
                      reduce_group_by="params",
                      reduce_output="model_selectionCV.csv")
        json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))
