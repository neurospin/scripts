"""
Created on Fri Feb 21 19:15:48 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import os
import json
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import nibabel
from sklearn.metrics import precision_recall_fscore_support
from parsimony.estimators import LogisticRegressionL1L2TV
import parsimony.functions.nesterov.tv as tv

##############################################################################
## User map/reduce functions
def A_from_structure(structure_filepath):
    # Input: structure_filepath. Output: A, structure
    STRUCTURE = nibabel.load(structure_filepath)
    A, _ = tv.A_from_mask(STRUCTURE.get_data())
    return A, STRUCTURE

def mapper(key, output_collector):
    import mapreduce  as GLOBAL # access to global variables:
    # GLOBAL.DATA, GLOBAL.STRUCTURE, GLOBAL.A
    # GLOBAL.DATA ::= {"X":[Xtrain, ytrain], "y":[Xtest, ytest]}
    # key: list of parameters
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
    mod = LogisticRegressionL1L2TV(k, l, g, GLOBAL.A, penalty_start=1, 
                                        class_weight="auto")
    mod.fit(GLOBAL.DATA["X"][0], GLOBAL.DATA["y"][0])
    y_pred = mod.predict(GLOBAL.DATA["X"][1])
    print "Time :",key,
    structure_data = GLOBAL.STRUCTURE.get_data() != 0
    arr = np.zeros(structure_data.shape)
    arr[structure_data] = mod.beta.ravel()[1:]
    beta3d = nibabel.Nifti1Image(arr, affine=GLOBAL.STRUCTURE.get_affine())
    ret = dict(y_pred=y_pred, y_true=GLOBAL.DATA["y"][1], beta3d=beta3d)
    output_collector.collect(key, ret)

def reducer(key, output_collectors):
    # key : string of intermediary key
    # load return dict correspondning to mapper ouput. they need to be loaded.
    print "load ", output_collectors
    values = [item.load("*.npy") for item in output_collectors]
    y_true = [item["y_true"].ravel() for item in values]
    y_pred = [item["y_pred"].ravel() for item in values]
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    #n_ite = np.mean([item["model"].algorithm.num_iter for item in values])
    n_ite = None
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores

if __name__ == "__main__":
    BASE = "/neurospin/brainomics/2013_adni/proj_classif_AD-CTL"
    #BASE = "/neurospin/tmp/brainomics/testenettv"
    WD = BASE.replace("/neurospin/brainomics", "/neurospin/tmp/brainomics")
    INPUT_DATA_X = os.path.join(WD, 'X_intercept.npy')
    INPUT_DATA_y = os.path.join(WD, 'y.npy')
    INPUT_MASK_PATH = os.path.join(WD, "mask.nii")
    NFOLDS = 5
    OUTPUT = os.path.join(WD, 'logistictvenet_intercept_5cv')
    if not os.path.exists(OUTPUT): os.makedirs(OUTPUT)

    #############################################################################
    ## Create dataset on /neurospin/tmp/brainomics
    ## ADD Intercept
    if False:
        X = np.load(os.path.join(BASE, 'X.npy'))
        X_inter = np.hstack((np.ones((X.shape[0], 1)), X))
        np.all(X_inter[:, 1:] == X)
        np.save(os.path.join(BASE, 'X_intercept.npy'), X_inter)
        X_inter = np.load(os.path.join(BASE,  'X_intercept.npy'))
        np.all(X_inter[:, 1:] == X)
        if not os.path.exists(WD): os.makedirs(WD)
        import shutil
        shutil.copyfile(os.path.join(BASE, 'X_intercept.npy'), os.path.join(WD, 'X_intercept.npy'))
        shutil.copyfile(os.path.join(BASE, 'y.npy'), os.path.join(WD, 'y.npy'))
        shutil.copyfile(os.path.join(BASE, "SPM", "template_FinalQC_CTL_AD", "mask.nii"),
                        os.path.join(WD, "mask.nii"))
        # sync data to gabriel
        os.system('rsync -azvu /neurospin/tmp/brainomics/2013_adni/proj_classif_AD-CTL gabriel.intra.cea.fr:/neurospin/tmp/brainomics/2013_adni/')
        # True

    #############################################################################
    ## Create config file
    y = np.load(INPUT_DATA_y)
    cv = [[tr.tolist(), te.tolist()] for tr,te in StratifiedKFold(y.ravel(), n_folds=5)]
    # parameters grid
    tv_range = np.arange(0, 1., .1)
    ratios = np.array([[1., 0., 1], [0., 1., 1], [.1, .9, 1], [.9, .1, 1], [.5, .5, 1]])
    alphas = [.01, .05, .1 , .5, 1.]
    l2l1tv =[np.array([[float(1-tv), float(1-tv), tv]]) * ratios for tv in tv_range]
    l2l1tv.append(np.array([[0., 0., 1.]]))
    l2l1tv = np.concatenate(l2l1tv)
    alphal2l1tv = np.concatenate([np.c_[np.array([[alpha]]*l2l1tv.shape[0]), l2l1tv] for alpha in alphas])
    # reduced parameters list
    #alphal2l1tv = alphal2l1tv[10:12, :]
    params = [params.tolist() for params in alphal2l1tv]
    # User map/reduce function file:
    try:
        user_func_filename = os.path.abspath(__file__)
    except:
        user_func_filename = os.path.join(os.environ["HOME"],
        "git", "scripts", "2013_adni", "proj_classif_AD-CTL", 
        "02_logistictvenet_intercept.py")
        print "USE", user_func_filename

    config = dict(data=dict(X=INPUT_DATA_X, y=INPUT_DATA_y),
                  params=params, resample=cv,
                  structure=INPUT_MASK_PATH,
                  map_output=os.path.join(OUTPUT, "results"),
                  user_func=user_func_filename,
                  ncore=12,
                  reduce_input=os.path.join(OUTPUT, "results/*/*"),
                  reduce_group_by=os.path.join(OUTPUT, "results/.*/(.*)"),
                  reduce_output=os.path.join(OUTPUT, "results.csv"))
                  
    json.dump(config, open(os.path.join(OUTPUT, "config.json"), "w"))

    #############################################################################
    print "# Start by running Locally with 2 cores, to check that everything os OK)"
    print "mapreduce.py --mode map --config %s/config.json --ncore 2" % OUTPUT
    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
    
    #############################################################################
    print "# Run on the cluster with 30 PBS Jobs"
    print "mapreduce.py --pbs_njob 30 --config %s/config.json" % OUTPUT
    
    #############################################################################
    print "# Reduce"
    print "mapreduce.py --mode reduce --config %s/config.json" % OUTPUT

"""
#Misc scipts

# Find number of accomplished tasks
find /neurospin/tmp/brainomics/2013_adni/proj_classif_AD-CTL/logistictvenet_intercept_5cv/results/ -name model.pkl|wc

# rm run files
find /neurospin/tmp/brainomics/2013_adni/proj_classif_AD-CTL/logistictvenet_intercept_5cv/results/0/ -name "*run*" |wc
# find run files
find /neurospin/tmp/brainomics/2013_adni/proj_classif_AD-CTL/logistictvenet_intercept_5cv/results/0/ -name "*run*" | while read f ; do rm $f ; done
""" 