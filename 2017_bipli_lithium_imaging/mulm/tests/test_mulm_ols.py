# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 09:13:14 2013

@author: edouard
"""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
import mulm
import statsmodels.api as sm


class TestMULMOLS(unittest.TestCase):

    def test_ttest(self):
        n = 100
        px = 5
        py_info = 2
        py_noize = 100
        beta = np.array([1, 0, -.5] + [0] * (px - 4) + [2]).reshape((px, 1))
        X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
        Y = np.random.randn(n, py_info + py_noize)
        # Causal model: add X on the first py_info variable
        Y[:, :py_info] += np.dot(X, beta)
        # Two-tailed t-test all the regressors
        contrasts = np.identity(X.shape[1])

        ## OLS with statmodels, need to iterate over Y columns
        sm_tvals = list()
        sm_pvals = list()
        for j in xrange(Y.shape[1]):
            mod = sm.OLS(Y[:, j], X)
            sm_ttest = mod.fit().t_test(contrasts)
            sm_tvals.append(sm_ttest.tvalue)
            sm_pvals.append(sm_ttest.pvalue)
        sm_tvals = np.asarray(sm_tvals).T
        sm_pvals = np.asarray(sm_pvals).T
        ## OLS with MULM two-tailed
        mod = mulm.MUOLS(Y, X).fit()
        mulm_tvals, mulm_pvals, mulm_df = mod.t_test(contrasts, pval=True, two_tailed=True)
        
        mod_block = mulm.MUOLS(Y, X).fit(block=True, max_elements=1000)
        mulm_tvals_block, mulm_pvals_block, mulm_df_block = mod_block.t_test(contrasts, pval=True, two_tailed=True)

        # Check that results are similar
        assert_almost_equal(mulm_tvals, sm_tvals)
        assert_almost_equal(mulm_pvals, sm_pvals)
        
        assert np.all(mulm_tvals == mulm_tvals_block)
        assert np.all(mulm_pvals == mulm_pvals_block)
        assert np.all(mulm_df == mulm_df_block)

#        ## OLS with MULM one-tailed
#        mod = mulm.MUOLS(Y, X).fit()
#        mulm_tvals, mulm_pvals, mulm_df = mod.t_test(contrasts, pval=True, two_tailed=False)
#
#        # Check that results ar similar
#        assert_almost_equal(mulm_tvals, sm_tvals)
#        sm_pvals /= 2
#        #sm_pvals[sm_pvals > 1] = 1
#        assert_almost_equal(mulm_pvals, sm_pvals)
#        mulm_pvals -  np.min(sm_pvals*2,1)


    def test_maxT(self):
        n = 100
        px = 5
        py_info = 2
        py_noize = 100

        beta = np.array([1, 0, -.5] + [0] * (px - 4) + [2]).reshape((px, 1))
        np.random.seed(42)
        X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))])
        Y = np.random.randn(n, py_info + py_noize)
        # Causal model: add X on the first py_info variable
        Y[:, :py_info] += np.dot(X, beta)
        contrasts = np.identity(X.shape[1])

        mod = mulm.MUOLS(Y, X).fit()
        tvals, rawp, df = mod.t_test(contrasts, pval=True, two_tailed=True)
        tvals2, maxT, df2 = mod.t_test_maxT(contrasts, two_tailed=True)
        assert np.all(tvals == tvals2)
        assert np.all(df == df2)
        
        mod_block = mulm.MUOLS(Y, X).fit(block=True, max_elements=1000)
        tvals_block, rawp_block, df_block = mod.t_test(contrasts, pval=True, two_tailed=True)
        tvals_block2, maxT_block, df_block2 = mod_block.t_test_maxT(contrasts, two_tailed=True)
        assert np.all(tvals_block == tvals_block2)
        assert np.all(tvals_block2 == tvals2)
        assert np.all(df_block == df_block2)
        assert np.all(df_block2 == df2)
        
        # More than 10 positive with uncorrected pval
        expected_tp = py_info * 3
        expected_fp = ((py_info + py_noize) * 5 - expected_tp) * 0.05
        expected_p = expected_tp + expected_fp
        # Test the number of rawp positive lie within a expected positive +-10
        assert (np.sum(rawp < 0.05) < (expected_p + 10)) and (np.sum(rawp < 0.05) > (expected_p - 10))
        assert (np.sum(rawp_block < 0.05) < (expected_p + 10)) and (np.sum(rawp_block < 0.05) > (expected_p - 10))

        # Test the number maxT positive lie within a expected true positive +-2
        assert np.sum(maxT < 0.05) < (expected_tp + 2) and np.sum(maxT < 0.05) > (expected_tp - 2)
        assert np.sum(maxT_block < 0.05) < (expected_tp + 2) and np.sum(maxT_block < 0.05) > (expected_tp - 2)

if __name__ == '__main__':

    unittest.main()
