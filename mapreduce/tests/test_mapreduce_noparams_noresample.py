# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:41:43 2014

@author: md238665

Test that we correctly treat the case where no parameters and no resampling are
given.
Copied from test_mapreduce_noparams.

MapReduce should stop before executing anything.

"""

import os
import json
import numpy as np
import tempfile

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
#    cv = [[tr.tolist(), te.tolist()] for tr, te in KFold(n, n_folds=2)]
    user_func_filename = os.path.abspath(__file__)

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            y="y.npy"),
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
