# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""


# Build a dataset: X, y, mask structure and cv
import os, glob
import json
import numpy as np

from sklearn.cross_validation import StratifiedKFold

INPUT_PATH = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"
INPUT_DATA_PATH = os.path.join(INPUT_PATH, '?.center.npy')
#y_PATH = os.path.join(INPUT_PATH, 'y.npy')
INPUT_MASK_PATH = os.path.join(INPUT_PATH, "SPM", "template_FinalQC_CTL_AD", "mask.nii")
NFOLDS = 5
CONFIG_5CV_PATH = os.path.join(INPUT_PATH, 'tv_5cv_center', 'config.json')
JOBS_5CV_PATH = os.path.join(INPUT_PATH, 'tv_5cv_center', 'jobs.json')
SRC_PATH = os.path.join(os.environ["HOME"], "git", "scripts", "2013_adni", "proj_classif_AD-CTL")
USER_FUNC_PATH = os.path.join(SRC_PATH, "userfunc_logistictvenet.center.py")

OUTPUT_5CV = os.path.join(INPUT_PATH, 'tv_5cv_center', 'cv')

#############################################################################
## Test on all DATA
if False:
    exec(open(USER_FUNC_PATH).read())
    
    def load_data(path_glob):
        filenames = glob.glob(path_glob)
        data = dict()
        for filename in filenames:
            key, _ = os.path.splitext(os.path.basename(filename))
            data[key] = np.load(filename)
        return data
    
    A, STRUCTURE = A_from_structure(INPUT_MASK_PATH)
    DATA = load_data(INPUT_DATA_PATH)
    
    Xtr = DATA['X.center']
    ytr = DATA['y.center']
    
    from parsimony.estimators import RidgeLogisticRegression_L1_TV
    from parsimony.algorithms.explicit import StaticCONESTA
    from parsimony.utils import LimitedDict, Info
    
    alpha, ratio_k, ratio_l, ratio_g = 0.05, .1, .1, .8
    time_curr = time.time()
    from parsimony.utils import Info
    
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = RidgeLogisticRegression_L1_TV(k, l, g, A, class_weight="auto",
                                        algorithm=StaticCONESTA(info=LimitedDict(Info.num_iter, Info.t)))
    
    mod.fit(Xtr, ytr)


#############################################################################
## BUILD CONFIG FILE
#X = np.load(X_PATH)
y = np.load(glob.glob(INPUT_DATA_PATH)[0])
cv = StratifiedKFold(y.ravel(), n_folds=NFOLDS)
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


# Save everything in a single file config.json
config = dict(data = INPUT_DATA_PATH, structure = INPUT_MASK_PATH,
              params = params, resample = cv,
              map_output = OUTPUT_5CV,
              job_file =JOBS_5CV_PATH, 
              user_func = USER_FUNC_PATH,
              cores = 8,
              reduce_input = OUTPUT_5CV + "/*/*",
              reduce_group_by = OUTPUT_5CV + "/.*/(.*)")

o = open(CONFIG_5CV_PATH, "w")
json.dump(config, o)
o.close()


# Use it
# ------
"""
# 1) Build jobs file ---
mapreduce.py --mode build_job --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json

# 2) Map ---
mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json --core 4

# 3) Reduce ---
mapreduce.py --mode reduce --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json --reduce_output  /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/results.csv

ssh $HOST_NS_FOUAD
cd /neurospin/brainomics/2013_adni/proj_classif_AD-CTL

ssh -t $HOST_NS_DESK    /home/ed203246/bin/mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json  --core 4
ssh -t $HOST_NS_FOUAD   /home/ed203246/bin/mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json  --core 4
ssh -t $HOST_NS_DIMITRI /home/ed203246/bin/mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json  --core 4
ssh -t $HOST_NS_DAVIDGO /home/ed203246/bin/mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json  --core 4
ssh -t $HOST_NS_JI      /home/ed203246/bin/mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_AD-CTL/tv_5cv_center/config.json  --core 2

ls -d tv_5cv_center/*/*_run*|while read f ; do echo rm -f $f ; done
"""

