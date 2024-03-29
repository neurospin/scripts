# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:22:21 2014

@author: edouard.duchesnay@cea.fr

Test of mapreduce with resampling and parameters passed as dict.

"""

import os
import json
import numpy as np
import tempfile
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import pandas as pd

from collections import OrderedDict
from itertools import product


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def resample(config, resample_key):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_key]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                             for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    Xtrain = GLOBAL.DATA_RESAMPLED["X"][0]
    Xtest = GLOBAL.DATA_RESAMPLED["X"][1]
    ytrain = GLOBAL.DATA_RESAMPLED["y"][0].ravel()
    ytest = GLOBAL.DATA_RESAMPLED["y"][1].ravel()
    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    y_pred = mod.fit(Xtrain, ytrain).predict(Xtest)
    output_collector.collect(key, dict(y_pred=y_pred, y_true=ytest))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict corresponding to mapper ouput. they need to be loaded.
    values = [item.load() for item in values.values()]
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    d = OrderedDict()
    d['r2'] = r2_score(y_true, y_pred)
    return d


if __name__ == "__main__":
    WD = tempfile.mkdtemp()
    ###########################################################################
    ## Create dataset
    np.random.seed(13031981)
    n, p = 50, 100
    X = np.random.rand(n, p)
    beta = np.random.rand(p, 1)
    y = np.dot(X, beta)
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)

    ###########################################################################
    ## Create config file
    # Resampling output should be named "fold_0", "fold_1", etc.
    cv = {"fold_" + str(i): [tr.tolist(), te.tolist()]
          for i, (tr, te) in enumerate(KFold(n, n_folds=2))}
    # Parameters output should be "param_0", "param_1", etc.
    params = {}
    for i, (alpha, l1_ratio) in enumerate(product([0.1, 1], [.1, .5, 1.])):
        key = "params_" + str(i)
        params[key] = [alpha, l1_ratio]
    user_func_filename = os.path.abspath(__file__)

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
                  params=params, resample=cv,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))
    exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                "..", "mapreduce.py"))
    ###########################################################################
    ## Apply map
    map_cmd = "%s -v --map %s/config.json" % (exec_path, WD)
    reduce_cmd = "%s -v --reduce %s/config.json" % (exec_path, WD)
    os.system(map_cmd)
    os.system(reduce_cmd)

    ###########################################################################
    ## Do it without mapreduce
    res = list()
    for _, key in params.items():
        # key = params[0]
        y_true = list()
        y_pred = list()
        for _, (tr, te) in cv.items():
            # tr, te = cv[0]
            Xtrain = X[tr, :]
            Xtest = X[te, :]
            ytrain = y[tr, :].ravel()
            ytest = y[te, :].ravel()
            mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
            y_pred.append(mod.fit(Xtrain, ytrain).predict(Xtest))
            y_true.append(ytest)
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        # As we reload mapreduce results, the params will be interpreted as
        # strings representation of tuples.
        # Here we apply the same representation
        res.append([str(tuple(key)), r2_score(y_true, y_pred)])
    true = pd.DataFrame(res, columns=["params", "r2"])
    mr = pd.read_csv(os.path.join(WD, 'results.csv'))
    m = pd.merge(true, mr, on="params", suffixes=["_true", "_mr"])
    assert np.allclose(m.r2_true, m.r2_mr)
