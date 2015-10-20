# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:55:30 2014

@author: cp243490
"""

import os
import json
import numpy as np
import tempfile
import pandas as pd
from mulm import MUOLSStatsCoefficients

from collections import OrderedDict


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.contrast = config["contrast"]


def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    perm = config["resample"][resample_nb]
    GLOBAL.DATA_PERM = {"Xp": GLOBAL.DATA["X"][perm, ...],
                        "Y": GLOBAL.DATA["Y"]}


def mapper(key, output_collector):
    import mapreduce as GLOBAL  # access to global variables
    Xp = GLOBAL.DATA_PERM["Xp"]
    Y = GLOBAL.DATA_PERM["Y"]
    contrast = GLOBAL.contrast
    muols = MUOLSStatsCoefficients()
    muols.fit(Xp, Y)
    tvals_perm, _, _ = muols.stats_t_coefficients(X=Xp,
                                                  Y=Y,
                                                  contrast=contrast,
                                                  pval=False)
    output_collector.collect(key, dict(tvals_perm=tvals_perm))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict corresponding to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    tvals_perm = np.concatenate([item["tvals_perm"].ravel()
                                 for item in values])
    d = OrderedDict()
    d['maxT'] = np.max(tvals_perm)
    return d


if __name__ == "__main__":
    WD = tempfile.mkdtemp()

    ###########################################################################
    ## Load dataset
    BASE_PATH = "/neurospin/brainomics/2014_deptms"
    MODALITY = "MRI"
    INPUT_CSI = os.path.join(BASE_PATH,          MODALITY)

    X = np.load(os.path.join(INPUT_CSI, "X_" + MODALITY + "_wb.npy"))
    y = np.load(os.path.join(INPUT_CSI, "y.npy"))
    Z = X[:, :3]
    Y = X[:, 3:]

    # y, intercept, age, sex
    DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
    DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
    DesignMat[:, 1] = 1  # intercept
    DesignMat[:, 2] = Z[:, 1]  # age
    DesignMat[:, 3] = Z[:, 2]  # sex

    np.save(os.path.join(WD, 'X.npy'), DesignMat)
    np.save(os.path.join(WD, 'Y.npy'), Y)

    #########################################################################
    ## Create config file
    nbperms = 10
    cv = [np.random.permutation(X.shape[0]).tolist()
          for i in range(nbperms)]
    params = [[0]]
    user_func_filename = os.path.abspath(__file__)

    contrast = [1, 0, 0, 0]

    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            Y="Y.npy"),
                  contrast=contrast,
                  params=params, resample=cv,
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_group_by="resample_index",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))
    exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                "..", "mapreduce.py"))

    #########################################################################
    ## Apply map
    map_cmd = "%s -v --map %s/config.json" % (exec_path, WD)
    reduce_cmd = "%s -v --reduce %s/config.json" % (exec_path, WD)
    os.system(map_cmd)
    os.system(reduce_cmd)

    #########################################################################
    ## Do it without mapreduce
    max_t = list()
    for perm in cv:
        Xp = DesignMat[perm, ]
        muols = MUOLSStatsCoefficients()
        muols.fit(Xp, Y)
        tvals_perm, _, df = muols.stats_t_coefficients(X=Xp,
                                                       Y=Y,
                                                       contrast=contrast,
                                                       pval=False)
        max_t.append(np.max(tvals_perm))

    true = pd.DataFrame.from_items([("permutation", range(nbperms)),
                                    ("maxT", max_t)])
    mr = pd.read_csv(os.path.join(WD, 'results.csv'))
    # Check same scores
    assert np.allclose(mr.maxT, true.maxT)
