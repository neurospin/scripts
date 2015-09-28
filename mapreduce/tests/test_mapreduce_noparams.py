# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:41:43 2014

@author: md238665

Test that we correctly treat the case where no parameters is given.
Copied from test_mapreduce.

The mapper function uses hardcoded values.

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

def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    Xtrain = GLOBAL.DATA_RESAMPLED["X"][0]
    Xtest = GLOBAL.DATA_RESAMPLED["X"][1]
    ytrain = GLOBAL.DATA_RESAMPLED["y"][0].ravel()
    ytest = GLOBAL.DATA_RESAMPLED["y"][1].ravel()
    mod = ElasticNet(alpha=1.0, l1_ratio=0.5)
    y_pred = mod.fit(Xtrain, ytrain).predict(Xtest)
    output_collector.collect(key, dict(y_pred=y_pred, y_true=ytest))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict corresponding to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    d = OrderedDict()
    d['r2'] = r2_score(y_true, y_pred)
    return d


if __name__ == "__main__":
    import mapreduce

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
    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=2)]
    user_func_filename = os.path.abspath(__file__)

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
                  resample=cv,
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
    y_true = list()
    y_pred = list()
    for tr, te in cv:
        # tr, te = cv[0]
        Xtrain = X[tr, :]
        Xtest = X[te, :]
        ytrain = y[tr, :].ravel()
        ytest = y[te, :].ravel()
        mod = ElasticNet(alpha=1.0, l1_ratio=0.5)
        y_pred.append(mod.fit(Xtrain, ytrain).predict(Xtest))
        y_true.append(ytest)
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_pred)
    # As we reload mapreduce results, the params will be interpreted as
    # strings representation of tuples.
    # Here we apply the same representation
    res.append([str(tuple()), r2_score(y_true, y_pred)])
    true = pd.DataFrame(res, columns=["params", "r2"])
    mr = pd.read_csv(os.path.join(WD, 'results.csv'))
    # Check same keys
    assert np.all(true.params == mr.params)
    m = pd.merge(true, mr, on="params", suffixes=["_true", "_mr"])
    # Check same scores
    assert np.allclose(m.r2_true, m.r2_mr)
