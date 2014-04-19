# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from  parsimony import datasets
import nibabel

WD = "/neurospin/tmp/brainomics/testenettv"
if not os.path.exists(WD): os.makedirs(WD)

user_func = """
print "LOAD user_func"
# "enettv_userfunc.py": user defined map/reduce functions
# -----------------------------------------------------
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import nibabel
from parsimony.estimators import RidgeLogisticRegression_L1_TV
import parsimony.functions.nesterov.tv as tv

def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    structure = nibabel.load(structure_filepath)
    A, _ = tv.A_from_mask(structure.get_data())
    print "A_from_structure", A
    return A, structure

def mapper(key, output_collector):
    # key: list of parameters
    # Glob. var.: DATA : dict of list(len == 2) of numpy arr.
    # Typically: {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto")
    mod.fit(DATA["X"][0], DATA["y"][0])
    y_pred = mod.predict(DATA["X"][1])
    print "Time :",key,
    structure_data = STRUCTURE.get_data() != 0
    arr = np.zeros(structure_data.shape)
    arr[structure_data] = mod.beta.ravel()
    beta3d = nibabel.Nifti1Image(arr, affine=STRUCTURE.get_affine())
    ret = dict(model=mod, y_pred=y_pred, y_true=DATA["y"][1], beta3d=beta3d)
    output_collector.collect(key, ret)

def reducer(key, values):
    # key : string of intermediary key
    # values: list of dict. list of all the values associated with intermediary key.
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]     
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = np.mean([len(item["model"].info["t"]) for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores
"""
of = open(os.path.join(WD, "enettv_userfunc.py"), "w")
of.writelines(user_func)
of.close()

#############################################################################
## Create dataset
n_samples, shape = 200, (100, 100, 1)
X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=5, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

# parameters grid
tv_range = np.arange(0, 1., .1)
ratios = np.array([[1., 0., 1], [0., 1., 1], [.1, .9, 1], [.9, .1, 1], [.5, .5, 1]])
alphas = [.01, .05, .1 , .5, 1.]
l2l1tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
l2l1tv.append(np.array([[0., 0., 1.]]))
l2l1tv = np.concatenate(l2l1tv)
alphal2l1tv = np.concatenate([np.c_[np.array([[alpha]]*l2l1tv.shape[0]), l2l1tv] for alpha in alphas])

alphal2l1tv = alphal2l1tv[10:12, :]

params = [params.tolist() for params in alphal2l1tv]

# Save X, y, mask structure and cv
np.save(os.path.join(WD, 'X.npy'), X)
np.save(os.path.join(WD, 'y.npy'), y)
nibabel.Nifti1Image(np.ones(shape), np.eye(4)).to_filename(os.path.join(WD, 'mask.nii'))

#############################################################################
## Create config file
cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=2)]

of = open(os.path.join(WD, "enettv_userfunc.py"), "w")
of.writelines(user_func)
of.close()
config = dict(data=dict(X=os.path.join(WD, "X.npy"),
                        y=os.path.join(WD, "y.npy")),
              params=params, resample=cv,
              structure = os.path.join(WD, 'mask.nii'),
              map_output=os.path.join(WD, "results"),
              user_func=os.path.join(WD, "enettv_userfunc.py"),
              ncore=2,
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
os.system("mapreduce.py --mode reduce --config config.json")

































import json
import numpy as np
from sklearn.cross_validation import KFold


# Build a dataset: X, y, mask structure and cv
import nibabel, json
from  parsimony import datasets
from sklearn.cross_validation import StratifiedKFold
n_samples, shape = 200, (100, 100, 1)
X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
shape=shape, snr=5, random_seed=1)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
cv = StratifiedKFold(y.ravel(), n_folds=2)
cv = [[tr.tolist(), te.tolist()] for tr,te in cv]

# parameters grid
tv_range = np.arange(0, 1., .1)
ratios = np.array([[1., 0., 1], [0., 1., 1], [.1, .9, 1], [.9, .1, 1], [.5, .5, 1]])
alphas = [.01, .05, .1 , .5, 1.]
l2l1tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
l2l1tv.append(np.array([[0., 0., 1.]]))
l2l1tv = np.concatenate(l2l1tv)
alphal2l1tv = np.concatenate([np.c_[np.array([[alpha]]*l2l1tv.shape[0]), l2l1tv] for alpha in alphas])

params = [params.tolist() for params in alphal2l1tv]

# Save X, y, mask structure and cv
np.save('X.npy', X)
np.save('y.npy', y)
nibabel.Nifti1Image(np.ones(shape), np.eye(4)).to_filename('mask.nii')

# Save everything in a single file config.json
config = dict(data = "?.npy", structure = 'mask.nii',
              params = params, resample = cv,
              map_output = "map_results",
              job_file = "jobs.json", 
              user_func = "logistictvenet_userfunc.py", # copy 
              cores = 8,
              reduce_input = "map_results/*/*",
              reduce_group_by = "map_results/.*/(.*)")

o = open("config.json", "w")
json.dump(config, o)
o.close()


print """
Copy or link logistictvenet_userfunc.py here, then Run:

# 1) Build jobs file ---
mapreduce.py --mode build_job --config config.json

# 2) Map ---
mapreduce.py --mode map --config config.json

# 3) Reduce ---
mapreduce.py --mode reduce --config config.json
"""