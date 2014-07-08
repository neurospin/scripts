# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 16:26:11 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 17:20:27 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import sys, os
import numpy as np
import time
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))



def webs_logr_weight(basepath, pw_name, precomp_save=True):
    precomp_save = False
    # 1- read constraints : we do not use Group Constraint here
#    from bgutils.build_websters import group_pw_snp2,get_websters_logr, pw_gene_snp2
#    fic = 'go_synaptic_snps_gene'  #'go_synaptic_snps_gene10'
#    group, group_names, snpList = group_pw_snp2(fic=fic, cache=True)
#    pw, _ = pw_gene_snp2(fic=fic, cache=True)
    from bgutils.build_websters import get_websters_logr
    from bgutils.utils_pw import build_msigdb
    group, group_names, pw, snpList = build_msigdb(
                 pw_name= pw_name, 
                 mask = os.path.join(basepath,'data','geno','genetic_control_xpt'), 
                 outdir=os.path.join(basepath,'data'), cache=True)
    
    # 2- get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_logr(snp_subset=snp_subset)

    # 3- fix X : add a ones constant regressor
    p = (X.shape)[1]                            # keep orig size
    X = np.hstack((np.ones((X.shape[0],1)),X))  # add intercept
    
    eps = 1e-6
    max_iter = 200
    conts = 20        # will be removed next version current max_iter x cont
     
    if precomp_save:        
        # 4- build A matrix
        #normalement ne sert a rien
        import parsimony.functions.nesterov.gl as gl
        weights = [np.sqrt(len(group[i])) for i in group]
        A = gl.A_from_groups(p, groups=group, weights=weights)
        
        import parsimony.algorithms.explicit as explicit
        import parsimony.estimators as estimators
        # 5- Logistic regresssion
        k = 0.002 #ridge 
        l = 0.0 #lasso ( if ENET k+l should be 1
        g = 0.0 
        logr_gl = estimators.RidgeLogisticRegression_L1_GL(
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
        logr_gl.fit(X, y )
        print "Fit duration : ", time.time() - stime
        print "================================================================="
    
    
        # 6- Interpretation
        beta = logr_gl.beta[1:] 
        np.savez(os.path.join(basepath,'data',pw_name+'-unbiased-beta'), beta)
    
    else:
        print "Now performing adaptive groupLasso"
        unbiased_beta = np.load(os.path.join(basepath,'data',pw_name+'-unbiased-beta.npz'))['arr_0']
        norme2 = np.linalg.norm(unbiased_beta)
        k = 1./norme2
        weights = [np.linalg.norm(unbiased_beta[group[i]]) for i in group]
#        weights = 1./np.asarray(weights)
        weights = 1./np.sqrt(np.asarray(weights))
        l = 0.
        g = 1.
        alpha = 50.
        
        import parsimony.functions.nesterov.gl as gl
        import parsimony.algorithms.explicit as explicit
        import parsimony.estimators as estimators

        A = gl.A_from_groups(p, groups=group, weights=weights)
        k, l, g = alpha * np.array((k, l , g))
        logr_gl = estimators.RidgeLogisticRegression_L1_GL(
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
        logr_gl.fit(X, y )
        print "Fit duration : ", time.time() - stime
        print "================================================================="
        
        beta = logr_gl.beta[1:] 
        mask = (logr_gl.beta[1:] != 0.).ravel()
        mask = (beta*beta>1e-8)
        from bgutils.pway_interpret import pw_status, pw_beta_thresh
        pw_status(pw, snpList, mask.ravel())
    
        # 7- 
        from bgutils.pway_plot import plot_pw
        beta = logr_gl.beta[1:].copy()
        beta = beta / np.max(np.abs(beta))
    #    nbeta = pw_beta_thresh(beta, threshold=1e-2)
    #    nbeta[nbeta!=0.] = 0.8
    #    nbeta[nbeta==0.] = 0.1
#        plot_pw(beta, pway=pw, snplist=snpList, cache=True)    
#        plt.show()
#    
#        plt.plot(logr_gl.info['f'], '+')
#        plt.show()
        return(dict(
            model=logr_gl,group=group, group_names=group_names, 
            pw=pw, snpList=snpList 
            ))


if __name__=="__main__":
    basepath = '/neurospin/brainomics/2013_brainomics_genomics/'
    pw_name= 'c2.cp.kegg.v3.1.symbols'
    pw_name = 'c7.go-synaptic.symbols'
    # Precompute weight
    webs_logr_weight(basepath, pw_name, precomp_save=True)
    # get fit
    #ret =  webs_logr_weight(basepath, pw_name, precomp_save=False)
