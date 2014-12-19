# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:58:26 2014

@author: cp243490
"""

import os
import json
import numpy as np
from mulm import MUOLS

from collections import OrderedDict


def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    GLOBAL.contrast = np.asarray(config["contrast"])


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
    muols = MUOLS(Y, Xp)
    muols.fit()
    tvals_perm, _, _ = muols.t_test(contrasts=contrast, pval=False,
                                            two_tailed=True)
    output_collector.collect(key, dict(tvals_perm=tvals_perm))


def reducer(key, values):
    # values are OutputCollectors containing a path to the results.
    # load return dict corresponding to mapper ouput. they need to be loaded.
    values = [item.load() for item in values]
    tvals_perm = np.concatenate([item["tvals_perm"].ravel()
                                    for item in values])
    tvals_perm = np.abs(tvals_perm)
    max_t = np.max(tvals_perm, axis=1)
    d = OrderedDict()
    d['tvals'] = max_t
    return d


if __name__ == "__main__":

    ###########################################################################
    ## Load dataset
    BASE_PATH = "/neurospin/brainomics/2014_imagen_fu2_adrs"
    DATA_PATH = os.path.join(BASE_PATH, "ADRS_datasets")
    X = np.load(os.path.join(DATA_PATH, "X.npy"))
    y = np.load(os.path.join(DATA_PATH, "y.npy"))
    Z = X[:, :3]
    Y = X[:, 3:]
    WD = os.path.join(BASE_PATH, "ADRS_univariate")
    if not os.path.exists(WD):
        os.makedirs(WD)
    # y, intercept, age, sex
    DesignMat = np.zeros((Z.shape[0], Z.shape[1] + 1))
    DesignMat[:, 0] = (y.ravel() - y.ravel().mean())  # y
    DesignMat[:, 1] = 1  # intercept
    DesignMat[:, 2] = Z[:, 1]  # age
    DesignMat[:, 3] = Z[:, 2]  # gender

    np.save(os.path.join(WD, 'X.npy'), DesignMat)
    np.save(os.path.join(WD, 'Y.npy'), Y)

    #########################################################################
    ## Create config file
    nbperms = 1000
    contrast = np.array([[1, 0, 0, 0]])

    cv = [np.random.permutation(y.shape[0]).tolist()
                                            for i in range(nbperms)]

    params = [[0]]
    user_func_filename = os.path.abspath(__file__)
    print user_func_filename
    # mapreduce will set its WD to the directory that contains the config file
    # use relative path
    config = dict(data=dict(X="X.npy",
                            Y="Y.npy"),
                  contrast=contrast.tolist(),
                  params=params, resample=cv,
                  map_output="results_tmax",
                  user_func=user_func_filename,
                  reduce_group_by="resample_index",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config_tmax.json"), "w"))
    exec_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      "..", "mapreduce.py"))

    #################################################################
    # Build utils files: sync (push/pull) and PBS
    import brainomics.cluster_gabriel as clust_utils
    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
        clust_utils.gabriel_make_sync_data_files(WD)
    cmd = "mapreduce.py --map  %s/config_tmax.json" % WD_CLUSTER
    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
    ################################################################
    # Sync to cluster
    print "Sync data to gabriel.intra.cea.fr: "
    os.system(sync_push_filename)

    ######################################################################
    print "# Start by running Locally with 2 cores, to check that everything is OK)"
    print "mapreduce.py --map %s/config.json --ncore 2" % WD
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    print "# 1) Log on gabriel:"
    print 'ssh -t gabriel.intra.cea.fr'
    print "# 2) Run one Job to test"
    print "qsub -I"
    print "cd %s" % WD_CLUSTER
    print "./job_Global_long.pbs"
    print "# 3) Run on cluster"
    print "qsub job_Global_long.pbs"
    print "# 4) Log out and pull Pull"
    print "exit"
    print sync_pull_filename
    #########################################################################
    print "# Reduce"
    print "mapreduce.py --reduce %s/config.json" % WD