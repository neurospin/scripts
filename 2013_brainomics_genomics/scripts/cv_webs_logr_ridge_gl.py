#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:57:55 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

"""
Ce fichier contient à la fois les mapper et reducer et la documenation pour 
faire de la cross validation.
"""
import numpy as np
import sys, os
import json
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
import time

# Build a dataset: X, y, mask structure and cv
def mapreduce_parameterize():
#    n_samples, shape = 200, (100, 100, 1)
#    base = "classif_%ix%ix%ix%i_" % tuple([n_samples] + list(shape))
#    X3d, y, beta3d, proba = datasets.classification.dice5.load(n_samples=n_samples,
#                                        shape=shape, snr=5, random_seed=1)
#    X = X3d.reshape((n_samples, np.prod(beta3d.shape)))

    base = 'grid'
    # 1- read constraints : we do not use Group Constraint here
    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))
    from bgutils.build_websters import group_pw_snp2,get_websters_logr, pw_gene_snp2
    group, group_names, snpList = group_pw_snp2(fic='go_synaptic_snps_gene10', cache=True)
    pw, _ = pw_gene_snp2(fic='go_synaptic_snps_gene10', cache=True)    
    
    # 2- get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_logr(snp_subset=snp_subset)
    #3 ajout regresseur 1
    X = np.hstack((np.ones((X.shape[0],1)),X))
    
    # create set of train test
    cv = StratifiedKFold(y.ravel(), n_folds=5)
    cv = [[tr.tolist(), te.tolist()] for tr,te in cv]
    
    # parameters grid
    alphas = [.01, .05, .1 , .5, 1.]
    #    couplage_kl_g = np.array([[1., 0., 1], [0., 1., 1], [.1, .9, 1], [.9, .1, 1], [.5, .5, 1]])
    couplage_kl_g = np.array([[1., 0., 1], [0., 0., 1], [.1, 0., 1], [.9, 0., 1], [.5, 0., 1]])
    #    k_range = np.arange(0, 1., .1)
    k_range = np.arange(0, 3., .5)
    alphas = [.01, .05, .1 , .5, 1.]
    l2l1k =[np.array([[float(1-k), float(1-k), k]]) * couplage_kl_g for k in k_range]
    l2l1k.append(np.array([[0., 0., 1.]]))
    l2l1k = np.concatenate(l2l1k)
    alphal2l1k = np.concatenate([np.c_[np.array([[alpha]]*l2l1k.shape[0]), l2l1k] for alpha in alphas])
    
    params = [params.tolist() for params in alphal2l1k]
    
    # Save X, y, mask structure and cv
    np.save(base+'X.npy', X)
    np.save(base+'y.npy', y)
#    nibabel.Nifti1Image(np.ones(shape), np.eye(4)).to_filename(base+'mask.nii')
    
    # Save everything in a single file config.json
    config = dict(data = "%s?.npy"%base, structure = base+'mask.nii',
        params = params, resample = cv,
        map_output = base + "map_results",
        job_file = base + "jobs.json", 
        user_func = "/home/vf140245/gits/scripts/2013_brainomics_genomics/scripts/cv_webs_logr_ridge_gl.py",
        cores = 8,
        reduce_input = base + "map_results/*/*",
        reduce_group_by = base + "map_results/.*/(.*)")
    
    o = open("config.json", "w")
    json.dump(config, o)
    o.close()


def A_from_structure(structure_filepath):
    """User defined function, to build the A matrices.

    Parameters
    ----------
    structure : string, filepath to the structure

    Return
    ------
    A, structure
    Those two objects will be accessible via global variables: A and STRUCTURE
    """
    sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))
    from bgutils.build_websters import group_pw_snp2
    #a terme on passe ici la fichier vers les données de contrainte
    group, group_names, snpList = group_pw_snp2(fic='go_synaptic_snps_gene', cache=True)
    tmp = []
    for i in group:
        tmp.extend(group[i])
    p = len(set(tmp))
    print "DEBUG: ",p
    import parsimony.functions.nesterov.gl as gl
    weights = [np.sqrt(len(group[i])) for i in group]
    A = gl.A_from_groups(p, groups=group, weights=weights)
    structure = None
    
    return A, structure


def mapper(key, output_collector):
    import parsimony.algorithms.explicit as explicit
    import parsimony.estimators as estimators
    
    base = 'grid'
#    print 'DBG> :', DATA.keys()
    Xtr = DATA["%sX"%base][0]
    Xte = DATA["%sX"%base][1]
    ytr = DATA["%sy"%base][0]
    yte = DATA["%sy"%base][1]

    # 5- Logistic regresssion
    alpha, ratio_k, ratio_l, ratio_g = key
    k, l, g = alpha *  np.array((ratio_k, ratio_l, ratio_g))
#    k = 0.9 #ridge 
#    l = 0.0 #lasso ( if ENET k+l should be 1
#    g = 5. 
    logr_gl = estimators.RidgeLogisticRegression_L1_GL(
                    k=k, l=l, g=g,
                    A=A,
                    output=True,
                    algorithm=explicit.StaticCONESTA(eps=1e-6,
                                                     continuations=20,
                                                     max_iter=200),
                    penalty_start=1,
                    mean=False)    #mean error of lST sq error
    time_curr = time.time()
    logr_gl.fit(Xtr, ytr )
    y_pred = logr_gl.predict(Xte)
    beta = logr_gl.beta
    print "Time :",key, ":", time.time() - time_curr, "ite:%i, time:%f" % (len(logr_gl.info["t"]), np.sum(logr_gl.info["t"]))
    time_curr = time.time()
    logr_gl.A = None
    ret = dict(model=logr_gl, y_pred=y_pred, y_true=yte, beta=beta)
    output_collector.collect(key, ret)

def reducer(key, values):
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




# parameters grid
if __name__=="__main__":
    if len(sys.argv)==1:
        print sys.argv[0], 'prepare | build | map | reduce'
    if sys.argv[1]=='prepare':
        mapreduce_parameterize()
    elif sys.argv[1]=='build':
        print "/home/vf140245/gits/brainomics-team/tools/mapreduce//mapreduce.py --mode build_job --config config.json"
    elif sys.argv[1]=='map':
        print "/home/vf140245/gits/brainomics-team/tools/mapreduce//mapreduce.py --mode map --config config.json"
    elif sys.argv[1]=='reduce':
        print "/home/vf140245/gits/brainomics-team/tools/mapreduce//mapreduce.py --mode reduce --config config.json"    
    else:
        print sys.argv[0], 'prepare | build | map | reduce'