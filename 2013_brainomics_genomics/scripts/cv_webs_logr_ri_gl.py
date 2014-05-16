#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:57:55 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

"""
Ce fichier contient à la fois les mapper et reducer et la documenation pour 
faire de la cross validation.
"""
import numpy as np
import sys, os
import json
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import time
import pickle
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))


##############################################################################
## User map/reduce functions
def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    # read weights infos    
    unbiased_beta = np.load(structure_filepath+'-unbiased-beta.npz')['arr_0']
    combo = pickle.load(open(structure_filepath+'.pickle'))
    group, group_names, pw, snpList = combo['group'], combo['group_names'],\
                                      combo['constraint'], combo['snpList']
    weights = [np.linalg.norm(unbiased_beta[group[i]]) for i in group]
    weights = 1./np.sqrt(np.asarray(weights))
    import parsimony.functions.nesterov.gl as gl
    A = gl.A_from_groups(len(snpList), groups=group, weights=weights)
    STRUCTURE = unbiased_beta
    
    return A, STRUCTURE

def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables:
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters    
    import parsimony.algorithms.explicit as explicit
    import parsimony.estimators as estimators
    eps = 1e-6
    max_iter = 200
    conts = 20        # will be removed next version current max_iter x cont

    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = estimators.RidgeLogisticRegression_L1_GL(
                        k=k, l=l, g=g,
                        A=GLOBAL.A,
                        output=True,
                        algorithm=explicit.StaticCONESTA(eps=eps,
                                                         continuations=conts,
                                                         max_iter=max_iter),
                        penalty_start=1,
                        mean=False)#(k, l, g, GLOBAL.A, class_weight="auto")
    mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["y"][0])
    y_pred = mod.predict(GLOBAL.DATA["X"][1])
    print "Time :",key,
#    structure_data = GLOBAL.STRUCTURE.get_data() != 0
#    arr = np.zeros(structure_data.shape)
#    arr[structure_data] = mod.beta.ravel()
#    beta3d = nibabel.Nifti1Image(arr, affine=GLOBAL.STRUCTURE.get_affine())
    ret = dict(model=mod, y_pred=y_pred, y_true=GLOBAL.DATA["y"][1])#, beta3d=beta3d)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # values are OutputCollerctors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]     
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
#    n_ite = np.mean([item["model"].algorithm.num_iter for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1])#, n_ite=n_ite)
    return scores

if __name__ == "__main__":
    WD = "/neurospin/tmp/brainomics/websters_logr_gl"
    if not os.path.exists(WD): os.makedirs(WD)

    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))
    from bgutils.build_websters import get_websters_logr

    #############################################################################
    # struct information
    basepath = '/neurospin/brainomics/2013_brainomics_genomics/'
    pwset_name = 'c7.go-synaptic.symbols'
    struct_and_weights_prefix = os.path.join(basepath,'data',pwset_name)

    #############################################################################
    ## Create dataset
    combo = pickle.load(open(struct_and_weights_prefix+'.pickle'))
    group, group_names, pw, snpList = combo['group'], combo['group_names'],\
                                      combo['constraint'], combo['snpList']
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_logr(snp_subset=snp_subset)
    # ajout regresseur 1
    X = np.hstack((np.ones((X.shape[0],1)),X))

    # Save X, y, mask structure and cv
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)

    
    # cv folds
    cv = StratifiedKFold(y.ravel(), n_folds=2)
    cv = [[tr.tolist(), te.tolist()] for tr,te in cv]
    
    # parameters grid
    unbiased_beta = np.load(struct_and_weights_prefix+'-unbiased-beta.npz')['arr_0']
    norme2 = np.linalg.norm(unbiased_beta)
    k = 1./norme2
    couplage_kl_g = np.array([[k, 0., 0.], [k, 0., 1.], [k, 0., 2.]])
    alphas = [25., 50.]
    alphal2l1k = np.concatenate([np.c_[np.array([[alpha]]*couplage_kl_g.shape[0]), couplage_kl_g] for alpha in alphas])
    # reduced parameters list
    params = [params.tolist() for params in alphal2l1k]

    # User map/reduce function file:
    user_func_filename = os.path.abspath(__file__)

    #############################################################################
    ## Create config file
    config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                            y=os.path.join(WD, "y.npy")),
                  params=params, resample=cv,
                  structure = struct_and_weights_prefix,
                  map_output=os.path.join(WD, "results"),
                  user_func=os.path.join(WD, user_func_filename),
                  ncore=6,
                  reduce_input=os.path.join(WD, "results/*/*"),
                  reduce_group_by=os.path.join(WD, "results/.*/(.*)"))
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    #############################################################################
    print "# Run Locally:"
    print "mapreduce.py --mode map --config %s/config.json" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    
    #############################################################################
    print "# Run on the cluster with 4 PBS Jobs"
    print "mapreduce.py --pbs_njob 4 --config %s/config.json" % WD
    
    #############################################################################
    print "# Reduce"
    print "mapreduce.py --mode reduce --config %s/config.json" % WD
