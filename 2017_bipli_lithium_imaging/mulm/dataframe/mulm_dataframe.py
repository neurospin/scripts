# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 12:09:43 2014

@author: edouard.duchesnay@gmail.com
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from collections import OrderedDict
from statsmodels.sandbox.stats.multicomp import multipletests

class MULM:
    """ Massive (application) of Univariate Linear Model on panda DataFrame.
    For all y in tagrets, for all x in regressors for all
    z in covar_models apply lm(y ~ x + z)

    Parameters
    ----------
    data: pandas DataFrame

    targets: list of strings, single columns of input dataframe,
        example ["y1", "y2"].

    regressors: list of string, single columns of input dataframe,
        example ["x1", "x2"].

    covar_models: list of string, formulae that apply on input dataframe,
        example ["z1", "z2", "z1+z2", "z1*z2"].

    out_filemane: string, output file of full statistics associate with the
    model. If None no such statistics are computed/stored. Warning if not
    None this slows dwn the execution speed.

    Return
    ------
    DataFrame with columns:
    "models", "target", "regressor", "covariate", "effect", "sd", "tvalue",
    "pvalue", "df".
    """

    def __init__(self, data, formulas, intercept=True):
        self.data = data
        self.formulas = formulas
        self.intercept = intercept
        #self.regressors = regressors
        #self.covar_models = covar_models
        #self.out_filemane = out_filemane

    def t_test(self, contrasts=None, out_filemane=None, anova=False):
        # Make sure contrasts is a list of list
        if contrasts is not None:
            if isinstance(contrasts, tuple):
                contrasts = list(contrasts)
            if not isinstance(contrasts, list):
                contrasts = [contrasts]
            def change(c):
                if isinstance(c, list): return c
                else: return [c]
            contrasts = [change(c) for c in contrasts]
        if out_filemane:
            out_fd = open(out_filemane,'w')
        #res_formulas, res_target, res_regressor, res_covariate = [], [], [], []
        res_formulas, res_target, res_contrasts = [], [], []
        res_effect, res_sd, res_tvalue, res_pvalue, res_df = [], [], [], [], []
        for formula in self.formulas:
            #formula = self.formulas[0]
            target, regressors = formula.split("~")
            # fit
            #dt = self.data[self.data[target].notnull()]
            y, X = dmatrices(formula, data=self.data, return_type='dataframe')
            if not self.intercept:
                X = X.ix[:, 1:]
            mod = sm.OLS(y, X)
            sm_fitted = mod.fit()
            if contrasts is not None and not anova:
                #contrasts = [1]
                if self.intercept:
                    contrasts_ = [[0] + c for c in contrasts]
                # pad with 0
                contrasts_ = [c + [0] * (X.shape[1] - len(c)) for c in contrasts_]
                #[0] + contrasts + [0] * (X.shape[1] - 2)
                contrasts_str = ["_".join(X.columns[np.where(c)[0]].tolist())
                    for c in contrasts_]
                # TODO FIXME IF CONTRASTS AS A MATRIX
                #contrasts_str = [X.columns[contrasts[i] != 0] for i in range(len(contrasts))]
            else:
                contrasts_ = np.identity(X.shape[1])
                contrasts_str = X.columns.tolist()
            #print contrasts_, formula, X.columns
            sm_ttest = sm_fitted.t_test(contrasts_)
            #
            res_formulas += [formula] * len(contrasts_)
            res_target += [target] * len(contrasts_)
            res_contrasts += contrasts_str
            #res_covariate += covars
            res_effect += sm_ttest.effect.ravel().tolist()
            res_sd += sm_ttest.sd.ravel().tolist()
            res_tvalue += sm_ttest.tvalue.ravel().tolist()
            res_pvalue += sm_ttest.pvalue.ravel().tolist()
            res_df += [sm_ttest.df_denom] * len(contrasts_)
            if out_filemane:
                out_fd.write("\n" + "=" * 78 + "\n")
                out_fd.write("== Model:" + formula + "\n")
                out_fd.write("=" * 78 + "\n")
                out_fd.write(sm_fitted.summary().as_text())
                out_fd.write("\n\n")
        if out_filemane:
            out_fd.close()
        o = OrderedDict()
        o["formula"] = res_formulas
        o["target"] = res_target
        o["contrast"] = res_contrasts
        o["effect"] = res_effect
        o["sd"] = res_sd
        o["tvalue"] = res_tvalue
        o["pvalue"] = res_pvalue
        o["df"] = res_df
        return pd.DataFrame(o)

    def t_test_maxT(self, contrasts, nperm=100):#:, alternative= "two_sided"):
        #alternatives = ["two_sided", "less", "greater", "one_sided_auto"]
        #if not alternative in alternatives:
        #    raise ValueError("Not a valid alternative")
        data_ori = self.data
        self.data = self.data.copy()
        stats = self.t_test(contrasts=contrasts, out_filemane=None)
        tmax = list()
        tmin = list()
        targets = set([formula.split("~")[0] for formula in self.formulas])
        for perm in range(nperm):
            print("Perm ", perm)
            # permut all targets
            for target in targets:
                self.data[target] = np.random.permutation(self.data[target])
            stats_p = self.t_test(contrasts=contrasts, out_filemane=None)
            tmax.append(np.max(stats_p.tvalue))
            tmin.append(np.min(stats_p.tvalue))
        self.data = data_ori
        self.tmax = np.array(tmax)
        self.tmin = np.array(tmin)
        self.tmax_2sided = np.max(np.abs(np.vstack([self.tmax, self.tmin])), axis=0)
        pvalues_twosided_maxT = np.asarray([np.sum(self.tmax_2sided >= np.abs(t)) / float(len(self.tmax))
                for t in stats.tvalue])
        pvalues_onesided_maxT = list()
        for t in stats.tvalue:
            if t >= 0:
                pvalues_onesided_maxT.append(np.sum(self.tmax >= t) / float(len(self.tmax)))
            else:
                pvalues_onesided_maxT.append(np.sum(self.tmin <= t) / float(len(self.tmin)))
        pvalues_onesided_maxT = np.asarray(pvalues_onesided_maxT)
        print("n pval two sided != one sided",
              np.sum(pvalues_onesided_maxT != pvalues_twosided_maxT))
        assert np.all(pvalues_onesided_maxT <= pvalues_twosided_maxT)
        stats["pvalues_onesided_maxT"] = pvalues_onesided_maxT
        stats["pvalues_twosided_maxT"] = pvalues_twosided_maxT
        stats["pvalue_fdr_bh"] = multipletests(stats.pvalue,
                                               method='fdr_bh')[1]
        return stats


if __name__ == "__main__":

    ###############
    # Build dataset
    ###############

    n = 100
    px_info = 5
    px_noise = 5
    regressors = ["x_%i" % i for i in range(px_info + px_noise)]

    pz = 3
    z_colnames = ["z_%i" % i for i in range(pz)]

    py_info = 2
    py_noise = 8
    targets = ["y_%i" % i for i in range(py_info + py_noise)]

    beta = np.concatenate([
        (np.arange(1, px_info+1) / float(px_info))[::-1],
        [0]*px_noise]).reshape((px_info + px_noise, 1))
    np.random.seed(1)
    X = np.random.randn(n, px_info + px_noise)
    Z = np.random.randn(n, pz)
    Y = np.random.randn(n, py_info + py_noise)
    # Causal model: add X on the first py_info variable
    Y[:, :py_info] += np.dot(X, beta) + np.dot(Z, [.1]*pz)[:, None] + 1.
    data = pd.DataFrame(np.concatenate([Y, X, Z], 1),
                        columns=targets + regressors + z_colnames)

    #######
    # Model
    #######
    covar_model = "+".join(z_colnames)
    formulas = ['%s~%s+%s' % (target, regressor, covar_model)
                for target in targets for regressor in regressors]
    model = MULM(data=data, formulas=formulas)

    #############################################
    # Statistics on regressors (of interest) only
    #############################################

    stats = model.t_test(contrasts=1, out_filemane=None)

    ntests = len(targets) *  len(regressors) * (1)
    assert stats.shape[0] == ntests
    # Check that P (Positive) P_expected-20% < P < P_expected-20%
    P = np.sum(stats.pvalue<0.05)
    TP = py_info * (px_info)
    ntests = len(targets) *  len(regressors) * (1) # no intercept
    P_expected = (ntests - TP) * .05 + TP
    #print P_expected, P
    assert (P_expected * 0.5 < P)  and (P < P_expected * 1.5)

    #########################################################"
    # Statistics on all regressors (of interest + covariates)
    #########################################################

    stats_full = model.t_test(out_filemane=None)
    # 1 + pz + 1 = regressor + covariates + intercept
    ntests = len(targets) *  len(regressors) * (1 + pz + 1)
    assert stats_full.shape[0] == ntests

    # Check that P (Positive) P_expected-20% < P < P_expected-20%
    P = np.sum((stats_full.pvalue<0.05) & (stats_full.contrast != 'Intercept'))
    TP = py_info * (px_info + pz)
    ntests = len(targets) *  len(regressors) * (1 + pz) # no intercept
    P_expected = (ntests - TP) * .05 + TP
    #print P_expected, P
    assert (P_expected * 0.5 < P)  and (P < P_expected * 1.5)


    ######################################
    # Tmax correction for mult comparisons
    ######################################
    model = MULM(data=data, formulas=formulas)
    self = model

    stats = model.t_test_maxT(contrasts=1, nperm=20)
    #stats = model.t_test_maxT(nperm=10, alternative="two_sided")
    # Check that P (Positive) P_expected-20% < P < P_expected-20%
    P = np.sum(stats.pvalues_onesided_maxT < 0.05)
    assert (P <= py_info * (px_info)) and (P > 1)

"""
run -i ~/git/pylearn-mulm/mulm/dataframe/mulm_dataframe.py


    def t_test2(self, full_model=False, out_filemane=None, anova=False):
        if out_filemane:
            out_fd = open(out_filemane,'w')
        res_formulas, res_target, res_regressor, res_covariate = [], [], [], []
        res_effect, res_sd, res_tvalue, res_pvalue, res_df = [], [], [], [], []
        for target in self.targets:
            for regressor in self.regressors:
                for covar_model in self.covar_models:
                    #print target, regressor, covar_model
                    if covar_model:
                        model = '%s~%s+%s' % (target, regressor, covar_model)
                    else:
                        model = '%s~%s' % (target, regressor)
                    # fit
                    dt = self.data[self.data[target].notnull()]
                    y, X = dmatrices(model, data=dt, return_type='dataframe')
                    mod = sm.OLS(y, X)
                    sm_fitted = mod.fit()
                    if not full_model and not anova:
                        contrasts = [[0, 1] + [0]*(X.shape[1]-2)]
                        covars = [covar_model]
                    else:
                        contrasts = np.identity(X.shape[1])
                        covars = X.columns.tolist()
                    sm_ttest = sm_fitted.t_test(contrasts)
                    #
                    res_formulas += [model] * len(contrasts)
                    res_target += [target] * len(contrasts)
                    res_regressor += [regressor] * len(contrasts)
                    res_covariate += covars
                    res_effect += sm_ttest.effect.ravel().tolist()
                    res_sd += sm_ttest.sd.ravel().tolist()
                    res_tvalue += sm_ttest.tvalue.ravel().tolist()
                    res_pvalue += sm_ttest.pvalue.ravel().tolist()
                    res_df += [sm_ttest.df_denom] * len(contrasts)
                    if out_filemane:
                        out_fd.write("\n" + "=" * 78 + "\n")
                        out_fd.write("== Model:" + model + "\n")
                        out_fd.write("=" * 78 + "\n")
                        out_fd.write(sm_fitted.summary().as_text())
                        out_fd.write("\n\n")
        if out_filemane:
            out_fd.close()
        o = OrderedDict()
        o["models"] = res_formulas
        o["target"] = res_target
        o["regressor"] = res_regressor
        o["covariate"] = res_covariate
        o["effect"] = res_effect
        o["sd"] = res_sd
        o["tvalue"] = res_tvalue
        o["pvalue"] = res_pvalue
        o["df"] = res_df
        return pd.DataFrame(o)
"""


