# -*- coding: utf-8 -*-
import os
import json
import tempfile
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from parsimony import datasets
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv_helper


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    STRUCTURE = nibabel.load(config["structure"])
    A, _ = tv_helper.A_from_mask(STRUCTURE.get_data())
    GLOBAL.A, GLOBAL.STRUCTURE = A, STRUCTURE


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    Xtrain = GLOBAL.DATA_RESAMPLED["X"][0]
    Xtest = GLOBAL.DATA_RESAMPLED["X"][1]
    ytrain = GLOBAL.DATA_RESAMPLED["y"][0]
    ytest = GLOBAL.DATA_RESAMPLED["y"][1]
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha * np.array((ratio_k, ratio_l, ratio_g))
    mod = LogisticRegressionL1L2TV(k, l, g, GLOBAL.A, class_weight="auto")
    y_pred = mod.fit(Xtrain, ytrain).predict(Xtest)
    ret = dict(model=mod, y_pred=y_pred, y_true=ytest, beta=mod.beta)
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
    WD = tempfile.mkdtemp()

    ###########################################################################
    ## Create dataset
    n_samples, shape = 50, (20, 20, 1)
    X3d, y, beta3d, proba = \
        datasets.classification.dice5.load(n_samples=n_samples,
                                           shape=shape, snr=5, random_seed=1)
    X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

    # Save X, y, mask structure and cv
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)
    nibabel.Nifti1Image(np.ones(shape),
                        np.eye(4)).to_filename(os.path.join(WD, 'mask.nii'))

    ###########################################################################
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr, te in StratifiedKFold(y.ravel(),
                     n_folds=2)]
    # parameters grid
    tv_range = [0.1, 0.5]
    ratios = np.array([[.6, .4, 1], [.4, .6, 1]])
    alphas = [.01, .1]
    l1l2tv = [np.array([[float(1 - tv), float(1 - tv), tv]]) * ratios
        for tv in tv_range]
    l1l2tv = np.concatenate(l1l2tv)
    alphal1l2tv = np.concatenate([np.c_[np.array([[alpha]]*l1l2tv.shape[0]), l1l2tv] for alpha in alphas])
    # reduced parameters list
    params = [params.tolist() for params in alphal1l2tv]
    # User map/reduce function file:
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
                  params=params, resample=cv,
                  structure='mask.nii',
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    ###########################################################################
    print "# Run Locally:"
    print "mapreduce.py --map --config %s/config.json" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)

    ###########################################################################
    print "# Reduce"
    print "mapreduce.py --reduce --config %s/config.json" % WD


