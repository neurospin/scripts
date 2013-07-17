# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:16:43 2013

@author: ed203246
"""

from epac.mulm import LinearRegression
# X(nxp), y(nx1), mask => MULMStats => pval(px1) => ClusterStat() => sizes(kx1)

from epac import BaseNode

class MULMStats(BaseNode):
    def transform(**Xy):
        lm = LinearRegression()
        lm.fit(X=Xy["Z"], Y=Xy["Y"])
        tval, pval = lm.stats(contrast=[], pval=True)
        Xy["pval"] = pval
        return Xy

class ClusterStats(BaseNode):
    def transform(**Xy):
        mask = Xy["mask"]
        pval = Xy["pval"]
        #...
        #clust_sizes = ...
        out = dict(clust_sizes=clust_sizes)
        return out

pipeline = Pipe(MULMStats(), ClusterStats())

pipeline.run(Y=X, Z=design_mat)
