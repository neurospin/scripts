# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 11:00:16 2016

@author: ad247405
"""
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
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
import brainomics.image_atlas
from scipy.stats import binom_test
from collections import OrderedDict
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
from collections import OrderedDict

BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_IMA_model_selection'
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV.xlsx")
#############################################################################


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.DATA_IMA = GLOBAL.load_data(config["data_IMA"])
    

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

    T_IMA = GLOBAL.DATA_IMA["X_IMA"] 
    y_IMA = GLOBAL.DATA_IMA["y_IMA"] 
    T = GLOBAL.DATA["X"]
    y = GLOBAL.DATA["y"]
    Tdiff=np.mean(T_IMA,axis=0)-np.mean(T[y==0],axis=0)
    T_IMA_diff=T_IMA-Tdiff
    
    Xtr=np.vstack((T_IMA_diff,Xtr))
    ytr=np.hstack((y_IMA,ytr))
     
    
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
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.coef_,  mask=mask)
    if output_collector:
        output_collector.collect(key, ret)
    else:
        return ret


       
def scores(key, paths, config, ret_y=False):
    import mapreduce
    print(key)
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    #prob_pred = [item["proba_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    #prob_pred = np.concatenate(prob_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    auc = roc_auc_score(y_true, y_pred) #area under curve score.

    betas = np.hstack([item["beta"] for item in values]).T
    import array_utils
    betas_t = np.vstack([array_utils.arr_threshold_from_norm2_ratio(betas[i, :], .99)[0] for i in range(betas.shape[0])])
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
        c = float(key[0])
        
        scores['c'] = c
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
    scores['param_key'] = key
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

    def argmaxscore_bygroup(data, groupby='fold', param_key="param_key", score="auc"):
        arg_max_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_max_byfold.append([fold, data_fold.ix[data_fold[score].argmax()][param_key], data_fold[score].max()])
        return pd.DataFrame(arg_max_byfold, columns=[groupby, param_key, score])

    print('## Refit scores')
    print('## ------------')
    byparams = groupby_paths([p for p in paths if not p.count("cvnested") and not p.count("refit/refit") ], 3) 
    byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}

    data = [list(byparams_scores[k].values()) for k in byparams_scores]

    columns = list(byparams_scores[list(byparams_scores.keys())[0]].keys())
    scores_refit = pd.DataFrame(data, columns=columns)
    
    print('## doublecv scores by outer-cv and by params')
    print('## -----------------------------------------')
    data = list()
    bycv = groupby_paths([p for p in paths if p.count("cvnested") and not p.count("refit/cvnested")  ], 1)
    for fold, paths_fold in bycv.items():
        print(fold)
        byparams = groupby_paths([p for p in paths_fold], 3)
        byparams_scores = {k:scores(k, v, config) for k, v in byparams.items()}
        data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
        scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)


    print('## Model selection')
    print('## ---------------')
    svm = argmaxscore_bygroup(scores_dcv_byparams); svm["method"] = "svm"
    
    scores_argmax_byfold = svm

    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_svm = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "refit", row["param_key"]) for index, row in svm.iterrows()], config)

   
    scores_cv = pd.DataFrame([["svm"] + list(scores_svm.values())], columns=["method"] + list(scores_svm.keys()))
   
         
    with pd.ExcelWriter(results_filename()) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_refit', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name='scores_cv', index=False)


##############################################################################


if __name__ == "__main__":
    BASE_PATH="/neurospin/brainomics/2016_classif_hallu_fmri_bis"
    WD = WD = '/neurospin/brainomics/2016_classif_hallu_fmri_bis/results_nov/multivariate_analysis/svm/with_IMA_model_selection'
    INPUT_DATA_X = os.path.join(BASE_PATH,'results26/multivariate_analysis/data','T.npy')
    INPUT_DATA_y = os.path.join(BASE_PATH,'results26/multivariate_analysis/data','y_state.npy')
    INPUT_DATA_subject = os.path.join(BASE_PATH,'results26/multivariate_analysis/data','subject.npy')   

    INPUT_DATA_IMA_X = os.path.join(BASE_PATH,'results26/multivariate_analysis/data','T_IMA.npy')
    INPUT_DATA_IMA_y = os.path.join(BASE_PATH,'results26/multivariate_analysis/data','y_IMA.npy')
    INPUT_MASK_PATH = os.path.join(BASE_PATH,'results26',"multivariate_analysis","MNI152_T1_3mm_brain_mask.nii.gz")
    INPUT_CSV = os.path.join(BASE_PATH,"population26oct.txt")

    pop = pd.read_csv(INPUT_CSV,delimiter=' ')
    number_subjects = pop.shape[0]
    NFOLDS_OUTER = number_subjects
    NFOLDS_INNER = 5

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    subject = np.load(INPUT_DATA_subject)
    
    #Outer loop
    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]     
    cv_outer.insert(0, None)   
    null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
    cv_outer[0] = null_resampling

    for cv_outer_i in range(1,number_subjects+1):
    
        test_bool=(subject==(cv_outer_i-1))
        train_bool=(subject!=(cv_outer_i-1))       
        cv_outer[cv_outer_i][0] = np.array([i for i, x in enumerate(train_bool) if x])
        cv_outer[cv_outer_i][1] =np.array([i for i, x in enumerate(test_bool) if x])

    import collections
    cv = collections.OrderedDict()
    for cv_outer_i, (tr_val, te) in enumerate(cv_outer):
        if cv_outer_i == 0:
            cv["refit/refit"] = [tr_val, te]
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["refit/cvnested%02d" % (cv_inner_i)] = [tr_val[tr], tr_val[val]]
        else:
            cv["cv%02d/refit" % (cv_outer_i -1)] = [tr_val, te]
            cv_inner = StratifiedKFold(y[tr_val].ravel(), n_folds=NFOLDS_INNER, random_state=42)
            for cv_inner_i, (tr, val) in enumerate(cv_inner):
                cv["cv%02d/cvnested%02d" % ((cv_outer_i-1), cv_inner_i)] = [tr_val[tr], tr_val[val]]
        
    for k in cv:
        cv[k] = [cv[k][0].tolist(), cv[k][1].tolist()]

       
    print(list(cv.keys()))  

  

    C_range = [[100],[10],[1],[1e-1],[1e-2],[1e-3],[1e-4],[1e-5],[1e-6],[1e-7],[1e-8],[1e-9]]
    #assert len(C_range) == 12
    
    
    user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2016_classif_hallu_fmri_bis", "svm_doubleCV_with_IMA.py")
    
    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  data_IMA=dict(X_IMA=INPUT_DATA_IMA_X, y_IMA=INPUT_DATA_IMA_y),
                  subject = INPUT_DATA_subject,
                  params=C_range, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output="model_selectionCV", 
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="params",
                  reduce_output="model_selectionCV.csv")
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))


