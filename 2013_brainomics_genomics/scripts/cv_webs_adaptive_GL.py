# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:52:36 2014
@author: fh235918
Created on Thu Jun  5 10:42:35 2014
@author: hl237680

Same as 15_cv_multivariate_residualized_BMI but with pathes taking into
account hl237680's username instead of vf140245 in order to send jobs on
Gabriel from my own account.
"""

import os, sys
import json
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support

import parsimony
import parsimony.estimators as estimators
import parsimony.functions.nesterov.gl as gl
import parsimony.algorithms.primaldual as explicit
from parsimony.utils.consts import Info

from bgutils.build_websters import group_pw_snp2,get_websters_logr

OUTPUT_BASE_DIR = "/neurospin/brainomics/2013_brainomics_genomics"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "tmp")
OUTPUT_SNP_FILE = os.path.join(OUTPUT_DIR, "X.npy")
OUTPUT_CLINIC_FILE =os.path.join(OUTPUT_DIR, "y.npy")

PENALTY_START = 1

def load_globals(config):
    import mapreduce as GLOBAL  # access to global variables
    GLOBAL.DATA = GLOBAL.load_data(config["data"])
    from bgutils.build_websters import group_pw_snp2
    fic = 'go_synaptic_snps_gene'  #'go_synaptic_snps_gene10'
    groups, group_names, snpList = group_pw_snp2(fic=fic, cache=True)
    GLOBAL.groups = groups

def resample(config, resample_nb):
    import mapreduce as GLOBAL  # access to global variables
    #GLOBAL.DATA = GLOBAL.load_data(config["data"])
    resample = config["resample"][resample_nb]
    print "reslicing %d" %resample_nb
    GLOBAL.DATA_RESAMPLED = {k: [GLOBAL.DATA[k][idx, ...] for idx in resample]
                            for k in GLOBAL.DATA}
    print "done reslicing %d" %resample_nb

    ###############################
    #weight computation for this fold
    ################################
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    p = Xtr.shape[1]
    groups = GLOBAL.groups
    # Compute A matrix (as penalty_start is 1 we need to only p-1 columns)
    Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_shape((p-PENALTY_START,))
    eps = 1e-6
    max_iter = 600
    info_conf = [Info.fvalue, Info.num_iter]
    logr_tv = estimators.LogisticRegressionL1L2TV(
                    l1=0, l2=0, tv=1, penalty_start=PENALTY_START,
                    A=Atv,
                    algorithm=explicit.StaticCONESTA(eps=eps,
                                                     max_iter=max_iter,
                                                     info=info_conf),
                    mean=False)
    logr_tv.fit(Xtr, ytr)
    beta_w = logr_tv.beta
    weights = [1./(np.linalg.norm(beta_w[group])) for group in groups]
    GLOBAL.weights = weights

def mapper(key, output_collector):
    import mapreduce as GLOBAL # access to global variables:
    # key: list of parameters
    k, l, g, alpha = key[0], key[1], key[2], key[3]
    Xtr = GLOBAL.DATA_RESAMPLED["X"][0]
    Xte = GLOBAL.DATA_RESAMPLED["X"][1]
    ytr = GLOBAL.DATA_RESAMPLED["y"][0]
    yte = GLOBAL.DATA_RESAMPLED["y"][1]
    #print key, "Data shape:", Xtr.shape, Xte.shape, ytr.shape, yte.shape

    p = Xtr.shape[1]
    groups = GLOBAL.groups
    weights = GLOBAL.weights
    eps = 1e-6
    max_iter = 600
    # Compute A matrix (we need only p-PENALTY_START columns)
    A_gl = gl.A_from_groups(p-PENALTY_START, groups=groups, weights=weights)
    mod = estimators.LogisticRegressionL1L2GL(
                    alpha*k, alpha*l, alpha*g,
                    A=A_gl,
                    algorithm=explicit.StaticCONESTA(eps=eps,
                                                     max_iter=max_iter),
                    penalty_start=PENALTY_START,
                    mean=False)      #since we residualized BMI with 2 categorical covariables (Gender and ImagingCentreCity - 8 columns) and 2 ordinal variables (tiv_gaser and mean_pds - 2 columns)
    y_pred = mod.fit(Xtr,ytr).predict(Xte)
    ret = dict(y_pred=y_pred, y_true=yte, beta=mod.beta)
    output_collector.collect(key, ret)


def reducer(key, values):
    # key : string of intermediary key
    # load return dict correspondning to mapper output. they need to be loaded.
    values = [item.load() for item in values]
    y_true = np.concatenate([item["y_true"].ravel() for item in values])
    y_pred = np.concatenate([item["y_pred"].ravel() for item in values])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    n_ite = np.mean([item["model"].algorithm.num_iter for item in values])
    scores = dict(key=key,
               recall_0=r[0], recall_1=r[1], recall_mean=r.mean(),
               precision_0=p[0], precision_1=p[1], precision_mean=p.mean(),
               f1_0=f[0], f1_1=f[1], f1_mean=f.mean(),
               support_0=s[0] , support_1=s[1], n_ite=n_ite)
    return scores


#"""
#run /home/hl237680/gits/scripts/2013_imagen_bmi/scripts/15_cv_multivariate_residualized_BMI.py
#"""
if __name__ == "__main__":

    ## Set pathes
    #WD = "/home/fh235918/git/scripts/2013_brainomics_genomics/test"
    WD = OUTPUT_DIR
    if not os.path.exists(WD): os.makedirs(WD)

    print '#############'
    print '# Read data #'
    print '#############'

    fic = 'go_synaptic_snps_gene'  #'go_synaptic_snps_gene10'
    group, group_names, snpList = group_pw_snp2(fic=fic, cache=True)

    # 2- get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    y, X_orig = get_websters_logr(snp_subset=snpList.tolist())

    # 3- fix X : add a ones constant regressor
    n, p = X_orig.shape  # keep orig size
    X = np.hstack((np.ones((X_orig.shape[0],1)),X_orig))  # add intercept

    print "#####################"
    print "# Build config file #"
    print "#####################"
    ## Parameterize the mapreduce
    ##   1) pathes
    np.save(OUTPUT_SNP_FILE, X)
    np.save(OUTPUT_CLINIC_FILE, y)
    ## 2) cv index and parameters to test
    NFOLDS = 5
    cv = [[tr.tolist(), te.tolist()] for tr,te in KFold(n, n_folds=NFOLDS)]
    params = [[k, l, g, alpha] for alpha in [1, 10] for l in np.arange(0.4, 1., .1)  for k in np.arange(0.4, 1., .1)  for g in np.arange(0.4, 1.)]
    # User map/reduce function file:
    try:
        user_func_filename = os.path.abspath(__file__)
    except:
        user_func_filename = os.path.join("/home/fh235918",
                                          "git", "scripts",
                                          "2013_brainomics_genomics",
                                          "scripts",
                                          "cv_webs_adaptive_GL.py")
    #print __file__, os.path.abspath(__file__)
    print "user_func", user_func_filename
    # Use relative path from config.json
    config = dict(data=dict(X=OUTPUT_SNP_FILE, y=OUTPUT_CLINIC_FILE),
                  params=params, resample=cv,
                  structure="",
                  map_output="results",
                  user_func=user_func_filename,
                  reduce_input="results/*/*",
                  reduce_group_by="results/.*/(.*)",
                  reduce_output="results.csv")
    json.dump(config, open(os.path.join(WD, "config.json"), "w"))

#    #############################################################################
#    # Build utils files: sync (push/pull) and PBS
#    sys.path.append(os.path.join(os.getenv('HOME'),
#                                'git','scripts'))
#    import brainomics.cluster_gabriel as clust_utils
#    sync_push_filename, sync_pull_filename, WD_CLUSTER = \
#        clust_utils.gabriel_make_sync_data_files(WD, user="hl237680")
#    cmd = "mapreduce.py -m %s/config.json  --ncore 12" % WD_CLUSTER
#    clust_utils.gabriel_make_qsub_job_files(WD, cmd)
#    #############################################################################
#    # Sync to cluster
#    print "Sync data to gabriel.intra.cea.fr: "
#    os.system(sync_push_filename)
#
#    #############################################################################
#    print "# Start by running Locally with 12 cores, to check that everything is OK)"
#    print "Interrupt after a while CTL-C"
#    print "mapreduce.py -m %s/config.json --ncore 12" % WD
#    #os.system("mapreduce.py --mode map --config %s/config.json" % WD)
#    print "# 1) Log on gabriel:"
#    print 'ssh -t gabriel.intra.cea.fr'
#    print "# 2) Run one Job to test"
#    print "qsub -I"
#    print "cd %s" % WD_CLUSTER
#    print "./job_Global_long.pbs"
#    print "# 3) Run on cluster"
#    print "qsub job_Global_long.pbs"
#    print "# 4) Log out and pull Pull"
#    print "exit"
#    print sync_pull_filename
#    #############################################################################
#    print "# Reduce"
#    print "mapreduce.py -r %s/config.json" % WD_CLUSTER
#    #ATTENTION ! Si envoi sur le cluster, modifier le path de config-2.json : /neurospin/tmp/hl237680/residual_bmi_images_cluster-2/config-2.json