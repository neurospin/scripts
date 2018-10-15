# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:25:41 2013

@author: ed203246
"""
import numpy as np
import scipy
from sklearn.preprocessing import scale
from scipy import stats

class MUPairwiseCorr:
    """Mass-univariate pairwise correlations. Given two arrays X [n_samples x p]
    and Y [n_samples x q]. Fit p x q independent linear models. Prediction
    and stats return [p x q] array.


    Example
    -------
    >>> import numpy as np
    >>> from mulm import MUPairwiseCorr
    >>> X = np.random.randn(10, 5)
    >>> Y = np.random.randn(10, 3)
    >>> corr = MUPairwiseCorr()
    >>> corr.fit(X, Y)
    <mulm.models.MUPairwiseCorr instance at 0x30da878>
    >>> f, p = corr.stats_f(X, Y)
    >>> print f.shape
    (5, 3)
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X, Y):
        Xs = scale(X, copy=True)
        Ys = scale(Y, copy=True)
        self.n_samples = X.shape[0]
        self.Corr_ = np.dot(Xs.T, Ys) / self.n_samples
        return self

    def predict(self, X):
        pass

    def stats_f(self, pval=True):
        R2 = self.Corr_ ** 2
        df_res = self.n_samples - 2
        f_stats = R2 * df_res / (1 - R2)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, 1, df_res)
            return f_stats, p_vals

class MUOLS:
    """Mass-univariate linear modeling based Ordinary Least Squares.
    Given two arrays X (n_samples, p) and Y (n_samples, q).
    Fit q independent linear models, ie., for all y in Y fit: lm(y ~ X)

    Example
    -------
    """

    def _block_slices(self, dim_size, block_size):
        """Generator that yields slice objects for indexing into
        sequential blocks of an array along a particular axis
        """
        count = 0
        while True:
            yield slice(count, count + block_size, 1)
            count += block_size
            if count >= dim_size:
                raise StopIteration

    def __init__(self, Y, X):
        self.coef = None
        if X.shape[0] != Y.shape[0]:
            raise ValueError('matrices are not aligned')
        self.X = X  # TODO PERFORM BASIC CHECK ARRAY
        self.Y = Y  # TODO PERFORM BASIC CHECK ARRAY

    def fit(self, block=False, max_elements=2 ** 27):
        """Use block=True for huge matrices Y.
        Operations block by block to optimize time and memory.
        max_elements: block dimension (2**27 corresponds to 1Go)
        """
        self.block = block
        self.max_elements = max_elements
        self.pinv = scipy.linalg.pinv2(self.X)
        n, p = self.Y.shape
        q = self.X.shape[1]
        if self.block:
            if self.max_elements < n:
                raise ValueError('the maximum number of elements is too small')
            max_cols = int(self.max_elements / n)
        else:
            max_cols = p
        self.coef = np.zeros((q, p))
        self.err_ss = np.zeros(p)
        for pp in self._block_slices(p, max_cols):
            if isinstance(self.Y, np.memmap):
                Y_block = self.Y[:, pp].copy()  # copy to force a read
            else: Y_block = self.Y[:, pp]
            #Y_block = self.Y[:, pp]
            self.coef[:, pp] = np.dot(self.pinv, Y_block)
            y_hat = np.dot(self.X, self.coef[:, pp])
            err = Y_block - y_hat
            del Y_block, y_hat
            self.err_ss[pp] = np.sum(err ** 2, axis=0)
            del err

#        self.coef = np.dot(self.pinv, self.Y)
#        y_hat = self.predict(self.X)
#        err = self.Y - y_hat
#        self.err_ss = np.sum(err ** 2, axis=0)
        return self

    def predict(self, X):
        #from sklearn.utils import safe_asarray
        import numpy as np
        #X = safe_asarray(X) # TODO PERFORM BASIC CHECK ARRAY
        pred_y = np.dot(X, self.coef)
        return pred_y


    def t_test(self, contrasts, pval=False, two_tailed=True):
        """Compute statistics (t-scores and p-value associated to contrast)

        Parameters
        ----------
        contrasts: The k contrasts to be tested, some list or array
        that can be casted to an k x p array.

        pval: boolean
            compute pvalues (default is false)

        two_tailed: boolean
            one-tailed test or a two-tailed test (default True)

        Return
        ------
        tstats (k, p) array, pvals (k, p) array, df (k,) array

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> X = np.random.randn(100, 5)
        >>> Y = np.random.randn(100, 10)
        >>> beta = np.random.randn(5, 1)
        >>> Y[:, :2] += np.dot(X, beta)
        >>> contrasts = np.identity(X.shape[1])
        >>> mod = mulm.MUOLS(Y, X).fit()
        >>> tvals, pvals, df = mod.t_test(contrasts, pval=True, two_tailed=True)
        """
        contrasts = np.atleast_2d(np.asarray(contrasts))
        n = self.X.shape[0]
        t_stats_ = list()
        p_vals_ = list()
        df_ = list()
        for contrast in contrasts:
            #ccontrasts = np.asarray(contrasts)
            # t = c'beta / std(c'beta)
            # std(c'beta) = sqrt(var_err (c'X+)(X+'c))
            #Xpinv = scipy.linalg.pinv(X)
            cXpinv = np.dot(contrast, self.pinv)
            R = np.eye(n) - np.dot(self.X, self.pinv)
            df = np.trace(R)
            ## Broadcast over ss errors
            var_errors = self.err_ss / df
            std_cbeta = np.sqrt(var_errors * np.dot(cXpinv, cXpinv.T))
            t_stats = np.dot(contrast, self.coef) / std_cbeta
            p_vals = None
            if pval is not None:
                if two_tailed:
                    p_vals = stats.t.sf(np.abs(t_stats), df) * 2
                else:
                    p_vals = stats.t.sf(t_stats, df)
            t_stats_.append(t_stats)
            p_vals_.append(p_vals)
            df_.append(df)
        return np.asarray(t_stats_), np.asarray(p_vals_), np.asarray(df_)

    def t_test_maxT(self, contrasts, nperms=1000, two_tailed=True, **kwargs):
        """Correct for multiple comparisons using maxT procedure. See t_test()
        For all parameters.

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> import pylab as plt
        >>> n = 100
        >>> px = 5
        >>> py_info = 2
        >>> py_noize = 100
        >>> beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
        >>> X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
        >>> Y = np.random.randn(n, py_info + py_noize)
        >>> Y[:, :py_info] += np.dot(X, beta)
        >>> contrasts = np.identity(X.shape[1])
        >>> mod = mulm.MUOLS(Y, X)
        >>> tvals, maxT, df = mod.t_test_maxT(contrasts, two_tailed=True)
        """
        #contrast = [0, 1] + [0] * (X.shape[1] - 2)
        contrasts = np.atleast_2d(np.asarray(contrasts))
        tvals, _, df = self.t_test(contrasts=contrasts, pval=False, **kwargs)
        max_t = list()
        for i in range(nperms):
            perm_idx = np.random.permutation(self.X.shape[0])
            Xp = self.X[perm_idx, :]
            muols = MUOLS(self.Y, Xp).fit(block=self.block,
                                          max_elements=self.max_elements)
            tvals_perm, _, _ = muols.t_test(contrasts=contrasts, pval=False,
                                            two_tailed=two_tailed)
            if two_tailed:
                tvals_perm = np.abs(tvals_perm)
            max_t.append(np.max(tvals_perm, axis=1))
            del muols
        max_t = np.array(max_t)
        tvals_ = np.abs(tvals) if two_tailed else tvals
        pvalues = np.array(
            [np.array([np.sum(max_t[:, con] >= t) for t in tvals_[con, :]])\
                / float(nperms) for con in range(contrasts.shape[0])])
        return tvals, pvalues, df

    def t_test_minP(self, contrasts, nperms=10000, two_tailed=True, **kwargs):
        """Correct for multiple comparisons using minP procedure.
        For all parameters.

        Example
        -------
        >>> import numpy as np
        >>> import mulm
        >>> import pylab as plt
        >>> n = 100
        >>> px = 5
        >>> py_info = 2
        >>> py_noize = 100
        >>> beta = np.array([1, 0, .5] + [0] * (px - 4) + [2]).reshape((px, 1))
        >>> X = np.hstack([np.random.randn(n, px-1), np.ones((n, 1))]) # X with intercept
        >>> Y = np.random.randn(n, py_info + py_noize)
        >>> Y[:, :py_info] += np.dot(X, beta)
        >>> contrasts = np.identity(X.shape[1])
        >>> tvals, maxT, df = mod.t_test_minP(contrasts, two_tailed=True)
        """
        contrasts = np.atleast_2d(np.asarray(contrasts))
        tvals, pvals, df = self.t_test(contrasts=contrasts, pval=True, **kwargs)
        min_p = np.ones((contrasts.shape[0], nperms))
        perm_idx = np.zeros((self.X.shape[0], nperms + 1), dtype='int')
        for i in range(self.Y.shape[1]):
            Y_curr = self.Y[:, i]
            Yp_curr = np.zeros((self.X.shape[0], nperms + 1))

            for j in range(nperms + 1):
                if i == 0:
                    perm_idx[:, j] = np.random.permutation(self.X.shape[0])
                Yp_curr[:, j] = Y_curr[perm_idx[:, j]]
            muols = MUOLS(Yp_curr, self.X).fit()
            tvals_perm, _, _ = muols.t_test(contrasts=contrasts, pval=False,
                                            two_tailed=two_tailed)
            if two_tailed:
                tvals_perm = np.abs(tvals_perm)
            pval_perm = np.array(
               [np.array([((np.sum(tvals_perm[con, :] >= tvals_perm[con, k])) - 1) \
                         for k in range(nperms)]) / float(nperms) \
                             for con in range(contrasts.shape[0])])
            min_p = np.array(
               [(np.min(np.vstack((min_p[con, :], pval_perm[con, :])), axis=0)) \
                         for con in range(contrasts.shape[0])])
        pvalues = np.array(
               [np.array([np.sum(min_p[con, :] <= p) \
                         for p in pvals[con, :]]) / float(nperms) \
                             for con in range(contrasts.shape[0])])
        return tvals, pvalues, df

    def f_test(self, contrast, pval=False):
        from sklearn.utils import array2d
        #Ypred = self.predict(self.X)
        #betas = self.coef
        #ss_errors = np.sum((self.Y - self.y_hat) ** 2, axis=0)
        C1 = array2d(contrast).T
        n, p = self.X.shape
        #Xpinv = scipy.linalg.pinv(X)
        rank_x = np.linalg.matrix_rank(self.pinv)
        C0 = np.eye(p) - np.dot(C1, scipy.linalg.pinv2(C1))  # Ortho. cont. to C1
        X0 = np.dot(self.X, C0)  # Design matrix of the reduced model
        X0pinv = scipy.linalg.pinv2(X0)
        rank_x0 = np.linalg.matrix_rank(X0pinv)
        # Find the subspace (X1) of Xc1, which is orthogonal to X0
        # The projection matrix M due to X1 can be derived from the residual
        # forming matrix of the reduced model X0
        # R0 is the residual forming matrix of the reduced model
        R0 = np.eye(n) - np.dot(X0, X0pinv)
        # R is the residual forming matrix of the full model
        R = np.eye(n) - np.dot(self.X, self.pinv)
        # compute the projection matrix
        M = R0 - R
        #Ypred = np.dot(self.X, betas)
        y_hat = self.predict(self.X)
        SS = np.sum(y_hat * np.dot(M, y_hat), axis=0)
        df_c1 = rank_x - rank_x0
        df_res = n - rank_x
        ## Broadcast over self.err_ss of Y
        f_stats = (SS * df_res) / (self.err_ss * df_c1)
        if not pval:
            return (f_stats, None)
        else:
            p_vals = stats.f.sf(f_stats, df_c1, df_res)
            return f_stats, p_vals

    def stats_f_coefficients(self, X, Y, contrast, pval=False):
        return self.stats_f(contrast, pval=pval)

