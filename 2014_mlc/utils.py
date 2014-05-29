# -*- coding: utf-8 -*-
"""
Created on Wed May 28 12:02:11 2014

@author: ed203246
"""

def mcnemar_test_prediction(y_pred1, y_pred2, y_true, cont_table=False):
    """Compute the mcnemar_test, return """
    import scipy.stats
    import numpy as np
    ok1 = y_pred1.ravel() == y_true.ravel()
    ok2 = y_pred2.ravel() == y_true.ravel()
    #a = np.sum(r1 & r2)
    a = np.sum(ok1 & ok2)
    d = np.sum(np.logical_not(ok1) & np.logical_not(ok2))
    b = np.sum(ok1 & np.logical_not(ok2))
    c = np.sum(np.logical_not(ok1) & ok2)
    #d = np.sum(np.logical_not(r1) & np.logical_not(r2))
    #mcnemar_chi2 = float((b -c) ** 2) / float(b + c)
    #The statistic with Yates's correction for continuity
    #mcnemar_chi2 = float((np.abs(b - c) - 1) ** 2) / float(b + c)
    #pval_chi2 = 1 - scipy.stats.chi2.cdf(mcnemar_chi2, 1)
    # If either b or c is small (b + c < 25) then \chi^2 is not well-approximated
    # by the chi-squared distribution
    # b is compared to a binomial distribution with size parameter equal to b + c
    # and "probability of success" = ½
    #b + c < 25, the binomial calculation should be performed, and indeed, most
    #software packages simply perform the binomial calculation in all cases,
    #since the result then is an exact test in all cases.
    #  to achieve a two-sided p-value in the case of the exact binomial test,
    # the p-value of the extreme tail should be multiplied by 2.
    pval_binom = np.minimum(scipy.stats.binom_test(b, n=b + c, p=0.5) * 2, 1.)
    if not cont_table:
        return pval_binom
    else:
        import pandas
        cont_table = pandas.DataFrame([[a,   b,     a + b],
                          [c,   d,     c + d],
                          [a+c, b + d, a + b + c + d ]],
        index=["1_Pos", "1_Neg", "Tot"], columns=["2_Pos", "1_Neg", "Tot"])
        return pval_binom, cont_table
