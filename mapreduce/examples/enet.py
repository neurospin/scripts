# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import tempfile
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from collections import OrderedDict

def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    Xtrain = GLOBAL.DATA_RESAMPLED["X"][0]
    Xtest = GLOBAL.DATA_RESAMPLED["X"][1]
    ytrain = GLOBAL.DATA_RESAMPLED["y"][0]
    ytest = GLOBAL.DATA_RESAMPLED["y"][1]
    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    y_pred = mod.fit(Xtrain, ytrain).predict(Xtest)
    output_collector.collect(key, dict(y_pred=y_pred, y_true=ytest))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict correspondning to mapper ouput. they need to be loaded.
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
    n, p = 100, 10000
    X = np.random.rand(n, p)
    beta = np.random.rand(p, 1)
    y = np.dot(X, beta)
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)

    ###########################################################################
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=2)]
    params = [[alpha, l1_ratio] for alpha in [0.01, 0.1, 1] for l1_ratio
        in np.arange(0.1, 1.1, .2)]
    user_func_filename = os.path.abspath(__file__)

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
                  params=params, resample=cv,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="params",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

    ###########################################################################
    print "# Run Locally:"
    print "mapreduce.py --map %s/config.json" % WD

    #############################################################################
    print "# Reduce"
    print "mapreduce.py --reduce %s/config.json" % WD
