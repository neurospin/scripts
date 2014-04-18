# -*- coding: utf-8 -*-
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