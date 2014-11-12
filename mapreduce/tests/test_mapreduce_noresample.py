# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:16:17 2014

@author: md238665

Test that we correctly treat the case where no resampling is given.
Copied from test_mapreduce.

The resample function is not needed. The mapper function uses the same dataset
for train and test.

"""

import os
import json
import numpy as np
import tempfile
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import pandas as pd

from collections import OrderedDict


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    X_train = GLOBAL.DATA["X"]
    X_test = GLOBAL.DATA["X"]
    y_train = GLOBAL.DATA["y"].ravel()
    y_test = GLOBAL.DATA["y"].ravel()
    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    y_pred = mod.fit(X_train, y_train).predict(X_test)
    output_collector.collect(key, dict(y_pred=y_pred, y_true=y_test))


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
    ## Create config file without resampling
    params = [[alpha, l1_ratio] for alpha in [0.1, 1] for l1_ratio
        in [.1, .5, 1.]]
    user_func_filename = os.path.abspath(__file__)

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
                  params=params,
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
    for key in params:
        # key = params[0]
        y_true = list()
        y_pred = list()
        X_train = X
        X_test = X
        y_train = y.ravel()
        y_test = y.ravel()
        mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
        y_pred.append(mod.fit(X_train, y_train).predict(X_test))
        y_true.append(y_test)
        y_true = np.hstack(y_true)
        y_pred = np.hstack(y_pred)
        # As we reload mapreduce results, the params will be interpreted as
        # strings representation of tuples.
        # Here we apply the same representation
        res.append([str(tuple(key)), r2_score(y_true, y_pred)])
    true = pd.DataFrame(res, columns=["params", "r2"])
    mr = pd.read_csv(os.path.join(WD, 'results.csv'))
    # Check same keys
    assert np.all(true.params == mr.params)
    m = pd.merge(true, mr, on="params", suffixes=["_true", "_mr"])
    # Check same scores
    assert np.allclose(m.r2_true, m.r2_mr)
