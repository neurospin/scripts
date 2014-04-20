# -*- coding: utf-8 -*-
import os
import json
import numpy as np
from sklearn.cross_validation import KFold

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

##############################################################################
## User map/reduce functions
user_func = """
# "enet_userfunc.py": user defined map/reduce functions
# -----------------------------------------------------

from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

def mapper(key, output_collector):
    # Global variables
    # ----------------
    # DATA : list(len == file matched by config["data"]) of list(len == 2) of numpy arr.
    # Typically: {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    mod = ElasticNet(alpha=key[0], l1_ratio=key[1])
    y_pred = mod.fit(DATA["X"][0], DATA["y"][0]).predict(DATA["X"][1])
    output_collector.collect(key, dict(y_pred=y_pred, y_true=DATA["y"][1]))


def reducer(key, values):
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    return dict(param=key, r2=r2_score(y_true, y_pred))
"""
of = open(os.path.join(WD, "enet_userfunc.py"), "w")
of.writelines(user_func)
of.close()

#############################################################################
## Create config file
cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=2)]
params = [[alpha, l1_ratio] for alpha in [0.01, 0.05, 0.1, 0.5, 1, 10] for l1_ratio in np.arange(0, 1.1, .1)]


config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                        y=os.path.join(WD, "y.npy")),
              params=params, resample=cv,
              map_output=os.path.join(WD, "results"),
              user_func=os.path.join(WD, "enet_userfunc.py"),
              ncore=12,
              reduce_input=os.path.join(WD, "results/*/*"),
              reduce_group_by=os.path.join(WD, "results/.*/(.*)"))
json.dump(config, open(os.path.join(WD, "config.json"), "w"))


#############################################################################
## Run Locally
os.system("mapreduce.py --mode map --config %s/config.json" % WD)

#############################################################################
## Or Run on the cluster with 4 PBS Jobs
## Execute messages
os.system("mapreduce.py --pbs_njob 4 --config %s/config.json" % WD)

#############################################################################
## 3) Reduce
os.system("mapreduce.py --mode reduce --config %s/config.json" % WD)
