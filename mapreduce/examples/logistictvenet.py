# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from  parsimony import datasets
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv

##############################################################################
## User map/reduce functions
def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    STRUCTURE = nibabel.load(structure_filepath)
    A, _ = tv.A_from_mask(STRUCTURE.get_data())
    return A, STRUCTURE

def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables:
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = LogisticRegressionL1L2TV(k, l, g, GLOBAL.A, class_weight="auto")
    mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["y"][0])
    y_pred = mod.predict(GLOBAL.DATA["X"][1])
    print "Time :",key,
    structure_data = GLOBAL.STRUCTURE.get_data() != 0
    arr = np.zeros(structure_data.shape)
    arr[structure_data] = mod.beta.ravel()
    beta3d = nibabel.Nifti1Image(arr, affine=GLOBAL.STRUCTURE.get_affine())
    ret = dict(model=mod, y_pred=y_pred, y_true=GLOBAL.DATA["y"][1], beta3d=beta3d)
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
    n_ite = np.mean([item["model"].algorithm.num_iter for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores

if __name__ == "__main__":
    WD = "/neurospin/tmp/brainomics/testenettv"
    if not os.path.exists(WD): os.makedirs(WD)

    #############################################################################
    ## Create dataset
    n_samples, shape = 200, (100, 100, 1)
    X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
    shape=shape, snr=5, random_seed=1)
    X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

    # Save X, y, mask structure and cv
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)
    nibabel.Nifti1Image(np.ones(shape), np.eye(4)).to_filename(os.path.join(WD, 'mask.nii'))

    #############################################################################
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=2)]
    # parameters grid
    tv_range = np.arange(0, 1., .1)
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.1, .9, 1], [.9, .1, 1], [.5, .5, 1]])
    alphas = [.01, .05, .1 , .5, 1.]
    l2l1tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l2l1tv.append(np.array([[0., 0., 1.]]))
    l2l1tv = np.concatenate(l2l1tv)
    alphal2l1tv = np.concatenate([np.c_[np.array([[alpha]]*l2l1tv.shape[0]), l2l1tv] for alpha in alphas])
    # reduced parameters list
    alphal2l1tv = alphal2l1tv[10:12, :]
    params = [params.tolist() for params in alphal2l1tv]
    # User map/reduce function file:
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                            y=os.path.join(WD, "y.npy")),
                  params=params, resample=cv,
                  structure = os.path.join(WD, 'mask.nii'),
                  map_output=os.path.join(WD, "results"),
                  user_func=os.path.join(WD, user_func_filename),
                  ncore=2,
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


