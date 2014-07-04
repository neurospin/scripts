# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 17:20:27 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys, os
import parsimony
import numpy as np
import time
import matplotlib.pyplot as plt
from parsimony.utils.consts import Info

sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))

if __name__=="__main__":
    # 1- read constraints : we do not use Group Constraint here
    from bgutils.build_websters import group_pw_snp2,get_websters_logr, pw_gene_snp2
    fic = 'go_synaptic_snps_gene'  #'go_synaptic_snps_gene10'
    groups, group_names, snpList = group_pw_snp2(fic=fic, cache=True)
    pw, _ = pw_gene_snp2(fic=fic, cache=True)

    # 2- get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X_orig = get_websters_logr(snp_subset=snp_subset)

    # 3- fix X : add a ones constant regressor
    p_orig = (X_orig.shape)[1]                            # keep orig size
    X = np.hstack((np.ones((X_orig.shape[0],1)),X_orig))  # add intercept
    p = (X.shape)[1]
    #

    # 4- build A matrix
    import parsimony.functions.nesterov.gl as gl
    import parsimony.algorithms.primaldual as explicit
    import parsimony.estimators as estimators
    Atv, n_compacts = parsimony.functions.nesterov.tv.A_from_shape((p_orig,))
    eps = 1e-8
    max_iter = 2600
    conts = 2        # will be removed next version current max_iter x cont
    info_conf = [Info.fvalue, Info.num_iter]
    logr_tv = estimators.LogisticRegressionL1L2TV(
                    l1=0, l2=0, tv=0.1,
                    A=Atv,
                    algorithm=explicit.StaticCONESTA(eps=eps,
                                                     max_iter=max_iter,
                                                     info=info_conf),
                    mean=False)
    logr_tv.fit(X_orig, y)
    beta_w = logr_tv.beta
#    plt.plot(beta_w[1:])
#    plt.show()


    PENALTY_START = 1
    extended_groups = groups 
#    + [[i] for i in range(PENALTY_START, p-1)]
    #test avec tv
    weights = [1./(np.linalg.norm(beta_w[group])) for group in extended_groups]
    #test avec lengeur
    #    weights = [np.sqrt(len(group[i])) for i in group]
    A = gl.A_from_groups(p-PENALTY_START, groups=extended_groups, weights=weights)

    # 5- Logistic regresssion
    eps = 1e-8
    max_iter = 2600
    conts = 2
    alpha=11       # will be removed next version current max_iter x cont
    k = (0.1)*(1./(np.linalg.norm(beta_w)))
    l = 0.1 #lasso ( if ENET k+l should be 1
    g =0.1
    logr_gl = estimators.LogisticRegressionL1L2GL(
                    l1=alpha*l, l2=alpha*k,  gl=alpha*g,
                    A=A,
                    algorithm=explicit.StaticCONESTA(eps=eps,
                                                     max_iter=max_iter),
                    penalty_start=1,
                    mean=False)    #mean error of lST sq error
    stime = time.time()
    print "================================================================="
    print "Now fitting the model"
    logr_gl.fit(X, y )
    print "Fit duration : ", time.time() - stime
    print "================================================================="


    # 6- Interpretation
    beta = logr_gl.beta[1:]
    mask = (logr_gl.beta[1:] != 0.).ravel()
#    mask = (beta*beta>1e-8)
    from bgutils.pway_interpret import pw_status, pw_beta_thresh
    pw_status(pw, snpList, mask.ravel())

    # 7-
    from bgutils.pway_plot import plot_pw
    beta = logr_gl.beta[1:].copy()
    beta = beta / np.max(np.abs(beta))
#    nbeta = pw_beta_thresh(beta, threshold=1e-2)
#    nbeta[nbeta!=0.] = 0.8
#    nbeta[nbeta==0.] = 0.1
    plot_pw(beta, pway=pw, snplist=snpList, cache=True)
    plt.show()

#    plt.plot(logr_gl.info['f'], '+')
#    plt.show()
