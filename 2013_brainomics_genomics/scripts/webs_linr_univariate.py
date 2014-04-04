# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:53:15 2014

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""

import sys, os
import numpy as np
sys.path.append(os.path.join(os.getenv('HOME'),
                                'gits','scripts','2013_brainomics_genomics'))


if __name__=="__main__":
    from bgutils.build_websters import get_websters_linr
    import mulm

    #get the snps list to get a data set w/ y continous variable
    #convenient snp order
    #subject order granted by the method
    snp_subset=['rs12120191', 'rs6687835'] 
    y, X = get_websters_linr(gene_name = 'KIF1B', snp_subset=snp_subset)
    n = y.shape[0]
    
    # select 2 snps of interest
    snp={}
    snp['rs12120191'] = 0
    snp['rs6687835']  = 1
    
    # R output
    #Build a unit test on that : meme chose sur plink rs6687835
    #> summary(lm(GEXP[,"GI_41393558.I"]~ms[,1]))
    #
    #Call:
    #lm(formula = Y[, 1] ~ X[, 1])
    #
    #Residuals:
    # Min   1Q  Median      3Q     Max 
    #-2.7101 -0.4352  0.0232  0.5453  2.1318 
    #
    #Coefficients:
    #Estimate Std. Error t value Pr(>|t|)
    #(Intercept) -0.01548 0.10158  -0.152 0.879
    #X[, 1] 0.01158 0.067840.171 0.865
    #
    #Residual standard error: 0.8629 on 362 degrees of freedom
    #Multiple R-squared:  8.051e-05, Adjusted R-squared:  -0.002682 
    #F-statistic: 0.02915 on 1 and 362 DF,  p-value: 0.8645    
    for k in snp.keys():
        print '\n====SNP : ', k
        x = X[:,snp[k]].reshape((n,-1))
        #transcoding should be performed to be compliant to R convention
        x[x==2]=3
        x[x==0]=2
        x[x==3]=0  
        #intercept
        x = np.hstack((x, np.ones((x.shape[0],1))))
        x[:,-1]=1
        olser = mulm.MUOLS()
        olser.fit(x, y)
        betas = olser.coef_
        contrast = [1., 0.]
        t, p = olser.stats_t_coefficients(x, y, contrast, pval=True)
        s, p = olser.stats_f_coefficients(x, y, contrast, pval=True)
        print 'betas = ',betas, '\n\n'
        print 'stat-t, stat-f,  p-val model additif= ',t, s, p
