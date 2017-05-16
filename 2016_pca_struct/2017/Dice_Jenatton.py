#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:11:00 2017

@author: ad247405
"""
import subprocess
import os
import json
import shutil
from collections import OrderedDict
import numpy as np
import sklearn.decomposition
from sklearn import metrics 
import parsimony.functions.nesterov.tv
import metrics
from brainomics import array_utils
import brainomics.cluster_gabriel as clust_utils
from parsimony.datasets.regression import dice5
import dice5_data
from sklearn.cross_validation import StratifiedKFold
import scipy.sparse as sparse
import parsimony.functions.nesterov.tv as nesterov_tv
import pca_struct
import parsimony.utils.check_arrays as check_arrays
################
# Input/Output #
################
INPUT_BASE_DIR = "/neurospin/brainomics/2016_pca_struct/dice"
INPUT_MASK_DIR = os.path.join(INPUT_BASE_DIR, "masks")
INPUT_DATA_DIR_FORMAT = os.path.join(INPUT_BASE_DIR,"data_0.1","data_{s[0]}_{s[1]}_{set}")
INPUT_STD_DATASET_FILE = "data.std.npy"
INPUT_OBJECT_MASK_FILE_FORMAT = "mask_{o}.npy"
INPUT_SNR_FILE = os.path.join(INPUT_BASE_DIR,"SNR.npy")
OUTPUT_BASE_DIR = os.path.join(INPUT_BASE_DIR, "2017","results_Jenatton")
OUTPUT_DIR_FORMAT = os.path.join(OUTPUT_BASE_DIR,"data_{s[0]}_{s[1]}_{set}")

##############
# Parameters #
##############

N_COMP = 10
NFOLDS_OUTER = 2
NFOLDS_INNER = 5


#############
# Functions #
#############


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    N_COMP = config["n_comp"]
    GLOBAL.N_COMP = N_COMP
    GLOBAL.OUTPUT_DIR = config["output_dir"]



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}



def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
    model_name, alpha, param = key 
    print (alpha)
    print (param)
    output_dir = GLOBAL.OUTPUT_DIR
    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    X_train_path = os.path.join(output_dir,output_collector.output_dir,"X_train.npy")
    np.save(X_train_path,X_train)
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
    N_COMP = GLOBAL.N_COMP
    output = os.path.join(output_dir,output_collector.output_dir)
    cmd = "matlab-R2017a -nodisplay -nojvm -r 'Jenatton_alpha " + X_train_path + " ("+str(N_COMP)+")"+" ("+str(alpha)+") "+" ("+str(param)+") "+ output + "/ ;quit'"
    os.chdir("/home/ad247405/Desktop/code_matlab/")
    os.system(cmd)
    
    
    V = np.load(os.path.join(output,"V.npy"))
    U_test,d = transform(X_test,V)
    U_train,d = transform(X_train,V)
    U_test_path = os.path.join(output_dir,output_collector.output_dir,"U_test.npy")
    U_train_path = os.path.join(output_dir,output_collector.output_dir,"U_train.npy")
    np.save(U_test_path,U_test)  
    np.save(U_train_path,U_train) 
    
    X_train_predict = predict(X_train,V)
    X_test_predict = predict(X_test,V)


    # Compute Frobenius norm between original and recontructed datasets
    frobenius_train = np.linalg.norm(X_train - X_train_predict, 'fro')
    frobenius_test = np.linalg.norm(X_test - X_test_predict, 'fro')

    # Remove predicted values (they are huge)
    del X_train_predict, X_test_predict
    frob_test_path = os.path.join(output_dir,output_collector.output_dir,"frob_test.npy")
    frob_train_path = os.path.join(output_dir,output_collector.output_dir,"frob_train.npy")
    np.save(frob_test_path,frobenius_test)  
    np.save(frob_train_path,frobenius_train) 
    


def scores(key, paths, config):
    import mapreduce
    values = [mapreduce.OutputCollector(p) for p in paths]
    values = [item.load() for item in values] 
    model = "Jenatton"
   

    frobenius_train = np.vstack([item["frob_train"] for item in values])
    frobenius_test = np.vstack([item["frob_test"] for item in values])
    components = np.dstack([item["V"] for item in values])


    #save mean pariwise across folds for further analysis   
#    dices_mean_path = os.path.join(WD,'dices_mean_%s.npy' %key.split("_")[0])
#    if key.split("_")[0] == 'struct_pca' and key.split("_")[2]<1e-3:
#        dices_mean_path = os.path.join(WD,'dices_mean_%s.npy' %"enet_pca")
#
#    np.save(dices_mean_path,dice_pairwise_values.mean(axis=0))

    
    #Mean frobenius norm across folds  
    mean_frobenius_train = frobenius_train.mean()
    mean_frobenius_test = frobenius_test.mean()

    print(key)
    scores = OrderedDict((
        ('param_key',key),
        ('model', model),
        ('frobenius_train', mean_frobenius_train),
        ('frobenius_test', mean_frobenius_test)))             
    return scores
    
def transform(X,V,in_place=False):
        """ Project a (new) dataset onto the components.
        Return the projected data and the associated d.
        We have to recompute U and d because the argument may not have the same
        number of lines.
        The argument must have the same number of columns than the datset used
        to fit the estimator.
        """
        Xk = check_arrays(X)
        if not in_place:
            Xk = Xk.copy()
        n, p = Xk.shape
        N_COMP = V.shape[1]
        if p != V.shape[0]:
            raise ValueError("The argument must have the same number of "
                             "columns than the datset used to fit the "
                             "estimator.")
        U = np.zeros((n, N_COMP))
        d = np.zeros((N_COMP))
        for k in range(N_COMP):
            # Project on component j
            vk = V[:, k].reshape(-1, 1)
            uk = np.dot(X, vk)
            uk /= np.linalg.norm(uk)
            U[:, k] = uk[:, 0]
            dk = compute_d(Xk, uk, vk)
            d[k] = dk
            # Residualize
            Xk -= dk * np.dot(uk, vk.T)
        return U,d
        
def compute_d(X, u, v):
        """Compute d that minimize the problem for u and v fixed.
           d = u^t.X.v / ||v||_2^2
        """
        norm_v2 = np.linalg.norm(v)**2
        d = np.dot(u.T, np.dot(X, v)) / norm_v2
        return d

def compute_rank1_approx(d, u, v):
        """Compute rank 1 approximation given by d, u, v.
           X_approx = d.u.v^t
        """
        X_approx = d * np.dot(u, v.T)
        return X_approx
        
def predict(X,V):
        """ Return the approximated matrix for a given matrix.
        We have to recompute U and d because the argument may not have the same
        number of lines.
        The argument must have the same number of columns than the datset used
        to fit the estimator.
        """
        Xk = check_arrays(X)
        n, p = Xk.shape
        N_COMP = V.shape[1]
        if p != V.shape[0]:
            raise ValueError("The argument must have the same number of "
                             "columns than the datset used to fit the "
                             "estimator.")
        Ut, dt = transform(X,V)
        Xt = np.zeros(Xk.shape)
        for k in range(N_COMP):
            vk = V[:, k].reshape(-1, 1)
            uk = Ut[:, k].reshape(-1, 1)
            Xt += compute_rank1_approx(dt[k], uk, vk)
        return Xt
        
def reducer(key, values):
    import os, glob, pandas as pd
    dir_path =os.getcwd()
    config_path = os.path.join(dir_path,"config_dCV.json")
    results_file_path = os.path.join(dir_path,"results_dCV_5folds_allparams.xlsx")
    config = json.load(open(config_path))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    paths = [p for p in paths if not p.count("Jenatton_9.313225746154785e-10")]


   
    def close(vec, val, tol=1e-4):
        return np.abs(vec - val) < tol

    def groupby_paths(paths, pos):
        groups = {g:[] for g in set([p.split("/")[pos] for p in paths])}
        for p in paths:
            groups[p.split("/")[pos]].append(p)
        return groups

    def argmaxscore_bygroup(data, groupby='fold', param_key="param_key", score="frobenius_test"):
        arg_min_byfold = list()
        for fold, data_fold in data.groupby(groupby):
            assert len(data_fold) == len(set(data_fold[param_key]))  # ensure all  param are diff
            arg_min_byfold.append([fold, data_fold.ix[data_fold[score].argmin()][param_key], data_fold[score].min()])
        return pd.DataFrame(arg_min_byfold, columns=[groupby, param_key, score])

    print('## Refit scores')
    print('## ------------')
    byparams = groupby_paths([p for p in paths if p.count("cv00/all") and not p.count("all/all")],3) 
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
        if fold == "cv01":
            byparams = groupby_paths([p for p in paths_fold], 3)
            byparams_scores = {k:scores(k, v, config) for k, v in list(byparams.items())}
            data += [[fold] + list(byparams_scores[k].values()) for k in byparams_scores]
            scores_dcv_byparams = pd.DataFrame(data, columns=["fold"] + columns)
        if fold == "cv00":
            print ("do not take into account second fold")

        
    print('## Model selection')
    print('## ---------------')
    folds= argmaxscore_bygroup(scores_dcv_byparams[scores_dcv_byparams["model"] == "Jenatton"])
    scores_argmax_byfold = folds
    

    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_jenatton = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "all", row["param_key"]) for index, row in folds.iterrows()], config)
    scores_cv = pd.DataFrame([["Jenatton"] + list(scores_jenatton.values())], columns=["method"] + list(scores_jenatton.keys()))
     
         
    with pd.ExcelWriter(results_file_path) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_all', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name = 'scores_cv', index=False)

#################
# Actual script #
#################

if __name__ == '__main__':
    # Read SNRs
    #input_snrs = np.load(INPUT_SNR_FILE)
    input_snrs=[0.1]
    # Resample 

    y = np.ones((500))
    cv_outer = [[tr, te] for tr,te in StratifiedKFold(y.ravel(), n_folds=NFOLDS_OUTER, random_state=42)]
    if cv_outer[0] is not None: # Make sure first fold is None
        cv_outer.insert(0, None)   
        null_resampling = list(); null_resampling.append(np.arange(0,len(y))),null_resampling.append(np.arange(0,len(y)))
        cv_outer[0] = null_resampling
                
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
    
    


########
    params = [('Jenatton',0.2,2**-29),('Jenatton',0.2,2**-28),('Jenatton',0.2,2**-27),('Jenatton',0.2,2**-26),('Jenatton',0.2,2**-25),('Jenatton',0.2,2**-24),('Jenatton',0.2,2**-23),('Jenatton',0.2,2**-22),\
              ('Jenatton',0.2,2**-21),('Jenatton',0.2,2**-20),('Jenatton',0.2,2**-19),('Jenatton',0.2,2**-18),('Jenatton',0.2,2**-17),\
              ('Jenatton',0.2,2**-16),('Jenatton',0.2,2**-15),('Jenatton',0.2,2**-14),('Jenatton',0.2,2**-13),('Jenatton',0.2,2**-12),\
              ('Jenatton',0.2,2**-11),('Jenatton',0.2,2**-10),('Jenatton',0.2,2**-0),('Jenatton',0.2,2**-8),('Jenatton',0.2,2**-7),\
              ('Jenatton',0.2,2**-6),('Jenatton',0.2,2**-5),('Jenatton',0.2,2**-4),('Jenatton',0.2,2**-3),('Jenatton',0.2,2**-2),\
              ('Jenatton',0.2,2**-1),('Jenatton',0.8,2**-29),('Jenatton',0.8,2**-28),('Jenatton',0.8,2**-27),('Jenatton',0.8,2**-26),('Jenatton',0.8,2**-25),('Jenatton',0.8,2**-24),('Jenatton',0.8,2**-23),('Jenatton',0.8,2**-22),\
              ('Jenatton',0.8,2**-21),('Jenatton',0.8,2**-20),('Jenatton',0.8,2**-19),('Jenatton',0.8,2**-18),('Jenatton',0.8,2**-17),\
              ('Jenatton',0.8,2**-16),('Jenatton',0.8,2**-15),('Jenatton',0.8,2**-14),('Jenatton',0.8,2**-13),('Jenatton',0.8,2**-12),\
              ('Jenatton',0.8,2**-11),('Jenatton',0.8,2**-10),('Jenatton',0.8,2**-0),('Jenatton',0.8,2**-8),('Jenatton',0.8,2**-7),\
              ('Jenatton',0.8,2**-6),('Jenatton',0.8,2**-5),('Jenatton',0.8,2**-4),('Jenatton',0.8,2**-3),('Jenatton',0.8,2**-2),\
              ('Jenatton',0.8,2**-1)]
#              
              
#    params = [('Jenatton',2**-29),('Jenatton',2**-28),('Jenatton',2**-27),('Jenatton',2**-26),('Jenatton',2**-25),('Jenatton',2**-24),('Jenatton',2**-23),('Jenatton',2**-22),\
#              ('Jenatton',2**-21),('Jenatton',2**-20),('Jenatton',2**-19),('Jenatton',2**-18),('Jenatton',2**-17),\
#              ('Jenatton',2**-16),('Jenatton',2**-15),('Jenatton',2**-14),('Jenatton',2**-13),('Jenatton',2**-12),\
#              ('Jenatton',2**-11),('Jenatton',2**-10),('Jenatton',2**-0),('Jenatton',2**-8),('Jenatton',2**-7),\
#              ('Jenatton',2**-6),('Jenatton',2**-5),('Jenatton',2**-4),('Jenatton',2**-3),('Jenatton',2**-2),\
#              ('Jenatton',2**-1)]

#    params = [('Jenatton',1e-6),(2**-25),(2**-24),(2**-23),(2**-22),(2**-21),(2**-20),\
#             (2**-19),(2**-18),(2**-17),(2**-16),(2**-15),\
#               (2**-14),(2**-13),(2**-12),(2**-1),(2**-10),(2**-9),(2**-8),\
#               (2**-7),(2**-6),(2**-5)]

    # Create a mapreduce config file for each dataset
    for set in range(0,50):
        #set=0.1
        input_dir = INPUT_DATA_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                 set=set)

        
        # Local output directory for this dataset
        output_dir = os.path.join(OUTPUT_BASE_DIR,
                                  OUTPUT_DIR_FORMAT.format(s=dice5_data.SHAPE,
                                                           set=set))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Copy the learning data
        src_datafile = os.path.join(input_dir, INPUT_STD_DATASET_FILE)
        shutil.copy(src_datafile, output_dir)

        # Copy the objects masks
        for i in range(3):
            filename = INPUT_OBJECT_MASK_FILE_FORMAT.format(o=i)
            src_filename = os.path.join(INPUT_MASK_DIR, filename)
            dst_filename = os.path.join(output_dir, filename)
            shutil.copy(src_filename, dst_filename)


        # Create config file
        user_func_filename = "/home/ad247405/git/scripts/2016_pca_struct/2017/Dice_Jenatton.py"

        config = OrderedDict([
            ('data', dict(X=INPUT_STD_DATASET_FILE)),
            ('output_dir',output_dir),
            ('resample', cv),
            ('params', params),
            ('n_comp', N_COMP),
            ('map_output', "results"),
            ('user_func', user_func_filename),
            ('reduce_group_by', "params"),
            ('reduce_output', "results.csv")])
        json.dump(config, open(os.path.join(output_dir, "config_alpha_dCV.json"), "w"))
        
        
#        
#            # Build utils files: sync (push/pull) and PBS
#        import brainomics.cluster_gabriel as clust_utils
#        sync_push_filename, sync_pull_filename, WD_CLUSTER = \
#            clust_utils.gabriel_make_sync_data_files(output_dir)
#        cmd = "mapreduce.py --map  %s/config_dCV.json" % WD_CLUSTER
#        clust_utils.gabriel_make_qsub_job_files(output_dir, cmd,walltime="200:00:00")
#    #        ################################################################
#        os.system(sync_push_filename)
