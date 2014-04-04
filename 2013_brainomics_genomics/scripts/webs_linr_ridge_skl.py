# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:00:32 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))


if __name__=="__main__":
    # read constraints : we do not use Group Constraint here
    from bgutils.build_websters import group_pw_snp,get_websters_linr
    group, group_names, snpList = group_pw_snp(nb=10)
    
    # get the snps list to get a data set w/ y continous variable
    # convenient snp order
    # subject order granted by the method
    snp_subset=np.asarray(snpList,dtype=str).tolist()
    y, X = get_websters_linr(gene_name = 'KIF1B', snp_subset=snp_subset)
    
    # center std 
    yc = y - y.mean()
    Xsd = (X - X.mean(axis=0))/X.std(axis=0)

    #    
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0, normalize=False)
    ridge.fit(Xsd, yc)
    
    #
    plt.plot(range(ridge.coef_.shape[0]), ridge.coef_)
    plt.show()