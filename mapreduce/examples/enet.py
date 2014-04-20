# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

##############################################################################
## User map/reduce functions
def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables (GLOBAL.DATA)
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    y_pred = mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["y"][0]).predict(GLOBAL.DATA["X"][1])
    output_collector.collect(key, dict(y_pred=y_pred, y_true=GLOBAL.DATA["y"][1]))


def reducer(key, values):
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    return dict(param=key, r2=r2_score(y_true, y_pred))


if __name__ == "__main__":
    WD = "/neurospin/tmp/brainomics/testenet"
    if not os.path.exists(WD): os.makedirs(WD)

    #############################################################################
    ## Create dataset
    n, p = 100, 10e4
    X = np.random.rand(n, p)
    beta = np.random.rand(p, 1)
    y = np.dot(X, beta)
    np.save(os.path.join(WD, 'X.npy'), X)
    np.save(os.path.join(WD, 'y.npy'), y)
    
    #############################################################################
    ## Create config file
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=2)]
    params = [[alpha, l1_ratio] for alpha in [0.01, 0.1, 1] for l1_ratio in np.arange(0, 1.1, .2)]
    user_func_filename = os.path.abspath(__file__)

    config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                            y=os.path.join(WD, "y.npy")),
                  params=params, resample=cv,
                  map_output=os.path.join(WD, "results"),
                  user_func=user_func_filename,
                  ncore=4,
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
