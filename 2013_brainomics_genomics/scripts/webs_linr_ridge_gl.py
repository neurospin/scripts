# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 18:10:48 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys, os
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))



if __name__=="__main__":
    # read constraints : we do not use Group Constraint here
    from bgutils.build_websters import group_pw_snp2,get_websters_linr, pw_gene_snp2
    group, group_names, snpList = group_pw_snp2(nb=10)
    pw, _ = pw_gene_snp2(nb=10)    
    
    # get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_linr(snp_subset=snp_subset)

    # fix X : add a ones constant regressor
    p = (X.shape)[1]                            # keep orig size
    X = np.hstack((np.ones((X.shape[0],1)),X))  # add intercept
    
    # build A matrix
    import parsimony.functions.nesterov.gl as gl
    weights = [np.sqrt(len(group[i])) for i in group]
    A = gl.A_from_groups(p, groups=group, weights=weights)


    import parsimony.algorithms.explicit as explicit
    import parsimony.estimators as estimators
    # linear regresssion
    eps = 1e-8
    max_iter = 200
    conts = 20        # will be removed next version current max_iter x cont
    k = 0.9 #ridge 
    l = 0.0 #lasso ( if ENET k+l should be 1
    g = 5. 
    linr_gl = estimators.RidgeRegression_L1_GL(
                    k=k, l=l, g=g,
                    A=A,
                    output=True,
                    algorithm=explicit.StaticCONESTA(eps=eps,
                                                     continuations=conts,
                                                     max_iter=max_iter),
                    penalty_start=1,
                    mean=False)    #mean error of lST sq error
    stime = time.time()
    print "================================================================="
    print "Now fitting the model"
    linr_gl.fit(X, y )
    print "Fit duration : ", time.time() - stime
    print "================================================================="

    #Interpretation
    mask = (linr_gl.beta[1:] != 0.).ravel()
    print "================================================================="
    print "nb feat selected : %d/%d"%(np.sum(mask), mask.shape[0])
    print "================================================================="
    sel_snp =set(snpList[mask])
    lig = ""
    for i,n in enumerate(pw):
        lig="\n==%s=\n"%str(n)
        for jg in pw[n]:
            lig += str(jg)+":%d/%d, "%(
                   len(set(pw[n][jg]).intersection(sel_snp)),
                   len(pw[n][jg]))  
        print lig
    
    
    plt.plot(linr_gl.beta[1:])
    plt.show()
    
    plt.plot(linr_gl.info['f'], '+')
    plt.show()
