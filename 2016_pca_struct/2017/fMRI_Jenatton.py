#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 17:50:46 2017

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
import array_utils
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
WD = '/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton'
NFOLDS_OUTER = 5
NFOLDS_INNER = 5
def config_filename(): return os.path.join(WD,"config_dCV.json")
def results_filename(): return os.path.join(WD,"results_dCV_5folds.xlsx")

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
    GLOBAL.spG_path = config["spG_path"]
    GLOBAL.OUTPUT_DIR = config["output_dir"]



def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}



def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables:
    model_name, param = key
    print (param)
    output_dir = GLOBAL.OUTPUT_DIR
    X_train = GLOBAL.DATA_RESAMPLED["X"][0]
    X_train_path = os.path.join(output_dir,output_collector.output_dir,"X_train.npy")
    np.save(X_train_path,X_train)
    X_test = GLOBAL.DATA_RESAMPLED["X"][1]
    N_COMP = GLOBAL.N_COMP
    spG_path = GLOBAL.spG_path;
    output = os.path.join(output_dir,output_collector.output_dir)
    cmd = "matlab-R2017a -nodisplay -nojvm -r 'Jenatton_3D " + X_train_path + " ("+str(N_COMP)+")"+" ("+str(param)+") "+ spG_path + " " + output + "/ ;quit'"
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
    comp = np.dstack([item["V"] for item in values])


    comp_t = np.zeros(comp.shape)
    comp_t_non_zero = np.zeros((10,5))
    for i in range(comp.shape[1]):
        for j in range(comp.shape[2]):
            comp_t[:,i,j] = array_utils.arr_threshold_from_norm2_ratio(comp[:,i,j], .99)[0]
            #comp_t_non_zero[i,j] = float(np.count_nonzero(comp_t[:,i,j]))/float(comp.shape[0])
            comp_t_non_zero[i,j] = float(np.count_nonzero(comp_t[:,i,j]))/float(63966)


    prop_non_zero_mean = np.mean(comp_t_non_zero[:,:])#do not count first comp
    print(prop_non_zero_mean)


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
        ('frobenius_test', mean_frobenius_test),
        ('prop_non_zeros_mean',prop_non_zero_mean)))
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
    results_file_path = os.path.join(dir_path,"results_dCV_5folds.xlsx")
    config = json.load(open(config_path))
    paths = glob.glob(os.path.join(config['map_output'], "*", "*", "*"))
    #paths = [p for p in paths if not p.count("sparse_pca_0.0_0.0_5.0")]


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

    #exclude when less than 1% of voxels are zero ---> too sparse
#    rm = (scores_dcv_byparams.prop_non_zeros_mean <0.01)
#    np.sum(rm)
#    scores_dcv_byparams = scores_dcv_byparams[np.logical_not(rm)]

    #exclude when less than 1% of voxels are zero ---> too sparse
    rm = (scores_dcv_byparams.prop_non_zeros_mean >0.50)
    np.sum(rm)
    scores_dcv_byparams = scores_dcv_byparams[np.logical_not(rm)]

    print('## Model selection')
    print('## ---------------')
    folds_je = argmaxscore_bygroup(scores_dcv_byparams[scores_dcv_byparams["model"] == "Jenatton"])
    scores_argmax_byfold = folds_je


    print('## Apply best model on refited')
    print('## ---------------------------')
    scores_je = scores("nestedcv", [os.path.join(config['map_output'], row["fold"], "all", row["param_key"]) for index, row in folds_je.iterrows()], config)

    scores_cv= pd.DataFrame([["enettv"] + list(scores_je.values())], columns=["method"] + list(scores_je.keys()))


    with pd.ExcelWriter(results_file_path) as writer:
        scores_refit.to_excel(writer, sheet_name='scores_all', index=False)
        scores_dcv_byparams.to_excel(writer, sheet_name='scores_dcv_byparams', index=False)
        scores_argmax_byfold.to_excel(writer, sheet_name='scores_argmax_byfold', index=False)
        scores_cv.to_excel(writer, sheet_name = 'scores_cv', index=False)

#################
# Actual script #
#################

if __name__ == "__main__":

    WD = '/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton'
    INPUT_DATA_X = '/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton/data/T_hallu.npy'
    INPUT_DATA_y = '/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton/data/y_hallu.npy'

    NFOLDS_OUTER = 5
    NFOLDS_INNER = 5
    NCOMP = 10


    #############################################################################
    ## Create config file
    ###########################
    #create DOUbleCV resampling pipeline
    y = np.load(INPUT_DATA_y)
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
    ###########################
    #Grid of parameters to explore
    ###########################
    params = [('Jenatton',2**-29),('Jenatton',2**-28),('Jenatton',2**-27),('Jenatton',2**-26),('Jenatton',2**-25),('Jenatton',2**-24),('Jenatton',2**-23),('Jenatton',2**-22),\
              ('Jenatton',2**-21),('Jenatton',2**-20),('Jenatton',2**-19),('Jenatton',2**-18),('Jenatton',2**-17),\
              ('Jenatton',2**-16),('Jenatton',2**-15),('Jenatton',2**-14),('Jenatton',2**-13),('Jenatton',2**-12),\
              ('Jenatton',2**-11),('Jenatton',2**-10),('Jenatton',2**-0),('Jenatton',2**-8),('Jenatton',2**-7),\
              ('Jenatton',2**-6),('Jenatton',2**-5),('Jenatton',2**-4),('Jenatton',2**-3),('Jenatton',2**-2),\
              ('Jenatton',2**-1)]


    user_func_filename = os.path.abspath(__file__)

    spG_path = "/neurospin/brainomics/2016_pca_struct/fmri/2017_fmri_Jenatton/data/spg.mat"
    config = OrderedDict([
            ('data', dict(X=INPUT_DATA_X)),
            ('output_dir',WD),
            ("spG_path",spG_path),
            ('resample', cv),
            ('params', params),
            ('n_comp', N_COMP),
            ('map_output', "results"),
            ('user_func', user_func_filename),
            ('reduce_group_by', "params"),
            ('reduce_output', "results.csv")])
    json.dump(config, open(os.path.join(WD, "config_dCV.json"), "w"))


