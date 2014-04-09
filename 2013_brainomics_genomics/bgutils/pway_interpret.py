# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:47:30 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""


import numpy as np
import sys, os
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))
    
def pw_status(pw, snpList, mask):
    print "================================================================="
    print "nb features selected : %d/%d"%(np.sum(mask), mask.shape[0])
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

def pw_beta_thresh(beta, threshold=1e-2):
    t = threshold**2
    betasq = beta.ravel()**2
    betasqsortindex = np.argsort(betasq)
    betasqcum = np.cumsum(betasq[betasqsortindex])     
    tmp = beta.copy()
    tmp[betasqsortindex[betasqcum < t]] = 0.
    
    return tmp

