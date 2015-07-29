# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:41:48 2014

@author: ed203246
"""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def mcnemar_test_classification(y_true, y_pred1, y_pred2, cont_table=False):
    """Compute the mcnemar_test for two classifiers.

    Input
    -----
        y_true: (n,) array, true labels

        y_pred1: (n,) array, first classifier predicted labels

        y_pred2: (n,) array, second classifier predicted labels

        cont_table: boolean, should the function return the contingency table

    Output
    ------
        p-value
    """
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



def auc_recalls_permutations_pval(y_true, y_pred1,  y_pred2, score_pred1, score_pred2, nperms=10000):
    """Compute the pvalues of recall_mean and auc for two classifiers.

    Input
    -----
        y_true: (n,) array, true labels

        y_pred1: (n,) array, first classifier predicted labels

        y_pred2: (n,) array, second classifier predicted labels

        score_pred1: (n,) array, first classifier predicted scores (for auc)

        score_pred2: (n,) array, second classifier predicted scores (for auc)
    Output
    ------
        pval_auc, pval_recall_mean
    """

    y_pred = np.c_[y_pred1, y_pred2]
    score_pred = np.c_[score_pred1, score_pred2]
    rows = np.arange(y_pred.shape[0])
    aucs = np.zeros(nperms + 1)
    recalls_mean = np.zeros(nperms + 1)
    
    for perm_i in xrange(nperms):
        if perm_i == 0:
            col2 = np.ones(y_pred.shape[0], dtype=int)
        else:
            col2 = np.random.randint(2, size=y_pred.shape[0])
        col1 = col2 - 1; col1 = col1 * np.sign(col1)
        assert np.all((col1 + col2) == 1)
        y_pred1 = y_pred[rows, col1]
        y_pred2 = y_pred[rows, col2]
        score_pred1 = score_pred[rows, col1]
        score_pred2 = score_pred[rows, col2]
        recalls1 = precision_recall_fscore_support(y_true, y_pred1, average=None)[1]
        auc1 = roc_auc_score(y_true, score_pred1) #area under curve score.
        recalls2 = precision_recall_fscore_support(y_true, y_pred2, average=None)[1]
        auc2 = roc_auc_score(y_true, score_pred2) #area under curve score.
        aucs[perm_i] = auc1 - auc2
        recalls_mean[perm_i] = recalls1.mean() - recalls2.mean()
    
    pval_auc = np.sum(aucs[1:] > aucs[0]) / float(nperms)
    pval_recall_mean = np.sum(recalls_mean[1:] > recalls_mean[0]) / float(nperms)
    return pval_auc, pval_recall_mean


def sign_permutation_pval(values, nperms=10000, stat="mean"):
    """Compute the pvalues of stat(values) vs stat(rand perm of sign values)

    Input
    -----
        values: (n,) array, numerical values

    Output
    ------
        pvals
    """
    stats = np.zeros(nperms + 1)
    for perm_i in xrange(nperms):
        if perm_i == 0:
            stats[perm_i] = np.mean(values)
        else:
            signs = np.random.randint(2, size=values.shape[0])
            signs[signs==0] = -1
            stats[perm_i] = np.mean(signs * values)
    return np.sum(stats[1:] > stats[0]) / float(nperms)
