# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 18:58:38 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import nibabel
from parsimony.estimators import RidgeLogisticRegression_L1_TV
import parsimony.functions.nesterov.tv as tv

def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    structure = nibabel.load(structure_filepath)
    A, _ = tv.A_from_mask(structure.get_data())
    return A, structure

def mapper(key, output_collector):
    # key: list of parameters
    # Glob. var.: DATA : dict of list(len == 2) of numpy arr.
    # Typically: {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto")
    mod.fit(DATA["X"][0], DATA["y"][0])
    y_pred = mod.predict(DATA["X"][1])
    print "Time :",key,
    structure_data = STRUCTURE.get_data() != 0
    arr = np.zeros(structure_data.shape)
    arr[structure_data] = mod.beta.ravel()
    beta3d = nibabel.Nifti1Image(arr, affine=STRUCTURE.get_affine())
    ret = dict(model=mod, y_pred=y_pred, y_true=DATA["y"][1], beta3d=beta3d)
    output_collector.collect(key, ret)

    
def reducer(key, values):
    # key : string of intermediary key
    # values: list of dict. list of all the values associated with intermediary key.
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]     
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = np.mean([len(item["model"].info["t"]) for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores
