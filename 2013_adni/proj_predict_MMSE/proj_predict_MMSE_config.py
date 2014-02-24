# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:50:14 2014

@author: md238665
"""

import numpy as np
import sklearn.linear_model

#
# Learning parameters
# Used in 01_ElasticNet.py)
#

N_FOLDS = 5
FOLD_PATH_FORMAT="{fold_index}"

# Random state for CV
#CV_SEED = 13031981
CV_SEED = 56
# Range of value for the global optimisation parameter
#  (alpha in ElasticNet)
#GLOBAL_PENALIZATION_RANGE = np.arange(0.1, 1, 0.1)
N_GLOBAL_PENALIZATION = 10

# Range of value for the l1 ratio parameter in ElasticNet
ENET_L1_RATIO_RANGE = [.1, .5, .7, .9, .95, .99, 1]

ENET_MODEL_PATH_FORMAT="{l1_ratio}-{alpha}"

#############
# UTILS
#############

def save_model(out_dir, mod, mask_im):
    import os, os.path, pickle, nibabel
    mask = mask_im.get_data() != 0
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pickle.dump(mod, open(os.path.join(out_dir, "model.pkl"), "w"))
    arr = np.zeros(mask.shape)
    arr[mask] = mod.beta.ravel()
    im_out = nibabel.Nifti1Image(arr, affine=mask_im.get_affine(), header=mask_im.get_header().copy())
    im_out.to_filename(os.path.join(out_dir,"beta.nii"))

#############
# Build CV to balance y test / train means
#############
#nfolds = proj_predict_MMSE_config.N_FOLDS

def BalancedCV(y, nfolds, random_seed=None):
    y = y.ravel()
    import itertools
    if random_seed is not None:  # If random seed, save current random state
        rnd_state = np.random.get_state()
    np.random.seed(random_seed)
    train = [list() for i in xrange(nfolds)]
    test = [list() for i in xrange(nfolds)]
    idx_sort = y.argsort()
    n = idx_sort.shape[0]
    idx_prev = 0
    for idx_cur in np.arange(nfolds, n,  nfolds):
        #idx_cur=5
        idx_set = idx_sort[idx_prev:idx_cur]
        #print items
        np.random.shuffle(idx_set)
        #print items
        idx_prev = idx_cur
        idx_idx_set = np.arange(idx_set.shape[0])
        for i in idx_idx_set:
            #print i
            test[i]  += idx_set[i == idx_idx_set].tolist()
            train[i] += idx_set[i != idx_idx_set].tolist()
    idx_left = idx_sort[idx_cur:n][::-1]
    # Left subject have higher score, will increase the mean of test fold
    # find smaller test -train and put left sample in those fold
    #[y[te, :].mean() - y[tr, :].mean() for tr, te in itertools.izip(train, test)]
    choosen_folds =np.argsort([y[te, :].mean() - y[tr, :].mean() for tr, te 
        in itertools.izip(train, test)])[:len(idx_left)]
    for i, fold in enumerate(choosen_folds):
        #print i, fold, idx_left[i] ,"in", fold
        for f in xrange(nfolds):
            if f == fold:
                test[f].append(idx_left[i])
            else:
                train[f].append(idx_left[i])
    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)
    return itertools.izip(train, test)

#for tr, te in itertools.izip(train, test):
#    print len(tr), len(te), y[tr, :].mean() - y[te, :].mean()
#
#CV = sklearn.cross_validation.KFold(
#    n,
#    shuffle=True,
#    #n_folds=2,
#    n_folds=proj_predict_MMSE_config.N_FOLDS)
#
#for tr, te in CV:
#    print len(tr), len(te), y[tr, :].mean() - y[te, :].mean() 
