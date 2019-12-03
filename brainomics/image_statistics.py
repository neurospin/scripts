#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/22/19

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import pandas as pd
import brainomics.image_preprocessing as preproc
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import mulm
import nilearn.image
import nilearn.plotting
import os.path
from matplotlib.backends.backend_pdf import PdfPages
import time
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

def univariate_statistics(NI_arr, ref_img, design_mat, pdf_filename, mask_arr=None, thres_nlpval=3):
    """
    Perform univariate statistics

    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    ref_img : image
    design_mat : DataFrame of the design matrix
    pdf_filename : str output pdf filename
    pdf : PdfPages(pdf_filename), optional
    mask_arr : ndarray of image_shape, if None use `preproc.compute_brain_mask(NI_arr, ref_img).get_data() > 0`
    thres_nlpval : float, threshold of neg log p-values

    Returns
    -------
        dict : of variables, where each variable is a dict of 'stat' and 'pval' computed on `mask_arr`.
            Use as `NI_arr[:, :, mask_arr].squeeze()`
        mask_arr : ndarray, the mask
    """

    pdf = PdfPages(pdf_filename)
    fig_title = os.path.splitext(os.path.basename(pdf_filename))[0]

    if mask_arr is None:
        mask_arr = preproc.compute_brain_mask(NI_arr, ref_img).get_data() > 0

    Y = NI_arr[:, :, mask_arr].squeeze()
    Y -= Y.mean(axis=0) # center Y to avoid capturing intercept

    ####################################################################################################################
    # build a variable dict with all required information, contrast for both numerical ad categorical variables

    #columns = X_df.columns
    Xdf_num = design_mat.select_dtypes('float')
    Xdf_cat = design_mat.select_dtypes('object')

    variables = collections.OrderedDict()
    i = 0
    for var in Xdf_num:
        variables[var] = dict(x=Xdf_num[var], idx=i, len=1, type='num')
        i += 1

    for var in Xdf_cat:
        x = pd.get_dummies(Xdf_cat[var])
        variables[var] = dict(x=x, idx=i, len=x.shape[1], type='cat')
        i += x.shape[1]

    # Design matrix concat numerical and categorical variables
    Xdf_dummy = pd.concat([variables[v]['x'] for v in variables], axis=1)

    if Xdf_cat.shape[1] > 0: # if not categorical variable, append intercept
        Xdf_dummy.assign(inter=1)

    for var in Xdf_num.columns:
        # print(var)
        con = np.zeros(Xdf_dummy.shape[1])
        con[variables[var]['idx']] = 1
        variables[var]['contrast'] = con

    for var in Xdf_cat.columns:
        # print(var)
        con = np.zeros((Xdf_dummy.shape[1], Xdf_dummy.shape[1]))
        indices = np.arange(variables[var]['idx'], variables[var]['idx'] + variables[var]['len'])
        con[indices, indices] = 1
        variables[var]['contrast'] = con

    X = np.asarray(Xdf_dummy)
    mod = mulm.MUOLS(Y, X)
    mod.fit()

    for var in Xdf_num.columns:
        # print(var)
        tvals, pvals, dof = mod.t_test(variables[var]['contrast'], pval=True, two_tailed=True)
        variables[var]['stat'] = tvals.squeeze()
        variables[var]['pval'] = pvals.squeeze()

    for var in Xdf_cat.columns:
        # print(var)
        fvals, pvals = mod.f_test(variables[var]['contrast'], pval=True)
        variables[var]['stat'] = fvals.squeeze()
        variables[var]['pval'] = pvals.squeeze()

    ####################################################################################################################
    # Plots

    # Plot brain maps
    fig, ax = plt.subplots(nrows=len(variables), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(variables):
        # print(cpt, var)
        ax_title = "%s: -log p-value (y~%s)" % (var ,"+".join(design_mat.columns))
        map_arr = np.zeros(mask_arr.shape)
        map_arr[mask_arr] = -np.log10(variables[var]['pval'])
        map_img = nilearn.image.new_img_like(ref_img, map_arr)
        nilearn.plotting.plot_glass_brain(map_img, colorbar=True, threshold=thres_nlpval, title=ax_title,
                                          figure=fig, axes=ax[cpt])

    if pdf:
        pdf.savefig()
    plt.close(fig)

    # Plot p-value histo
    fig, ax = plt.subplots(nrows=len(variables), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(variables):
        # print(cpt, var)
        ax_title = "%s: histo p-value (y~%s)" % (var, "+".join(design_mat.columns))
        ax[cpt].hist(variables[var]['pval'], bins=100)
        ax[cpt].set_title(ax_title)

    if pdf:
        pdf.savefig()
    plt.close(fig)

    df = design_mat.copy()
    df["gd_mean"] = Y.mean(axis=1)

    # variable x subject grand mean
    fig, ax = plt.subplots(nrows=len(variables), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(variables):
        # print(cpt, var)
        if (variables[var]['type'] == 'cat') or (len(df[var].unique()) <= 2):
            if "sex" in df and False:
                sns.violinplot(x=var, y="gd_mean", hue="sex", data=df, ax=ax[cpt])
            else:
                sns.violinplot(x=var, y="gd_mean", data=df, ax=ax[cpt])
        else:
            sns.scatterplot(x=var, y="gd_mean", data=df, ax=ax[cpt])

    if pdf:
        pdf.savefig()
    plt.close(fig)

    if pdf:
        pdf.close()

    # Clean variables
    for var in variables:
        variables[var] = dict(stat=variables[var]['stat'], pval=variables[var]['pval'])

    return variables, mask_arr

def ml_predictions(NI_arr, y, estimators, cv, mask_arr=None):
    """

    Parameters
    ----------
    NI_arr : ndarray, of shape (n_subjects, 1, image_shape).
    y : (n, ) array
        the target
    estimators : dict of estimators (do not mix regressors with classifiers)
    cv : cross-val
    mask_arr : ndarray of image_shape `NI_arr[:, :, mask_arr].squeeze()`

    Returns
    -------
    stats_df (DataFrame) of predictions statistics average across folds
    stats_folds_df (DataFrame) of predictions statistics for each individuals folds
    model_params dictionary of models parameters
    """
    if mask_arr is not None:
        X = NI_arr[:, :, mask_arr].squeeze()
    else:
        X = NI_arr

    estimator_type = set([e._estimator_type for e in estimators.values()]).pop()

    if cv is None:
        if estimator_type == 'classifier':
            cv = StratifiedKFold(n_splits=5)
        else:
            cv = KFold(n_splits=5)

    from sklearn.externals.joblib import Parallel, delayed
    from sklearn.base import is_classifier, clone

    def _split_fit_predict(estimators, X, y, train, test, scale=True):
        X_train, X_test, y_train, y_test = X[train, :], X[test, :], y[train], y[test]
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        results = {name:dict() for name in estimators}
        for name in estimators:
            t0 = time.clock()
            estimators[name].fit(X_train, y_train)
            results[name]["time"] = time.clock() - t0
            results[name]["y_test"] = estimators[name].predict(X_test)
            try:
                results[name]["score_test"] = estimators[name].decision_function(X_test)
            except:
                pass
            try:
                results[name]["coefs"] = estimators[name].coef_
            except:
                pass
        return results

    #estimator = lm.LogisticRegression(C=1, solver='lbfgs')

    parallel = Parallel(n_jobs=5)
    cv_ret = parallel(
        delayed(_split_fit_predict)(
            {name:clone(estimators[name]) for name in estimators}, X, y, train, test)
        for train, test in cv.split(X, y))

    # Aggregate predictions
    preds = {name:dict(y_test=np.zeros(len(y)), score_test=np.zeros(len(y)), time=[], coefs=[]) for name in estimators}
    for i, (train, test) in enumerate(cv.split(X, y)):
        for name in estimators:
            preds[name]['y_test'][test] = cv_ret[i][name]['y_test']
            try:
                preds[name]['score_test'][test] = cv_ret[i][name]['score_test']
            except:
                if 'score_test' in preds[name]: preds[name].pop('score_test')
            preds[name]['time'].append(cv_ret[i][name]['time'])
            try:
                preds[name]["coefs"].append(cv_ret[i][name]['coefs'].ravel())
            except:
                if 'coefs' in preds[name]: preds[name].pop('coefs')

    # Compute statistics
    #stats = {name:dict() for name in estimators}
    stats_list = list()
    stats_folds_list = list()

    if estimator_type == 'classifier':
        for name in preds:
                accs_test = np.asarray([metrics.accuracy_score(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
                baccs_test = np.asarray([metrics.recall_score(y[test], preds[name]['y_test'][test], average=None).mean() for train, test in cv.split(X, y)])
                recalls_test = np.asarray([metrics.recall_score(y[test], preds[name]['y_test'][test], average=None) for train, test in cv.split(X, y)])
                aucs_test = np.asarray([metrics.roc_auc_score(y[test], preds[name]['score_test'][test]) for train, test in cv.split(X, y)])

                stats_folds_list.append(
                    np.concatenate((np.array([name] * len(baccs_test))[: ,None],
                                   np.array(["CV%i" %cv for cv in range(len(baccs_test))])[: ,None],
                                   baccs_test[:, None], aucs_test[:, None], np.asarray(recalls_test),
                                   np.asarray(preds[name]['time'])[:, None]), axis=1))
                stats_list.append([name, np.mean(baccs_test), np.mean(aucs_test)] + \
                                      np.asarray(recalls_test).mean(axis=0).tolist() + [np.mean(preds[name]['time'])])

        stats_df = pd.DataFrame(stats_list, columns=['model', 'bacc_test', 'auc_test'] + \
                                                    ['recall%i_test' % lev for lev in np.unique(y)] + ["time"])

        stats_folds_df = pd.DataFrame(np.concatenate(stats_folds_list, axis=0), columns=['model', 'fold', 'bacc_test', 'auc_test'] + \
                                                    ['recall%i_test' % lev for lev in np.unique(y)] + ["time"])

    else:
        for name in preds:
            mae_test = np.asarray([metrics.mean_absolute_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            r2_test = np.asarray([metrics.r2_score(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            mse_test = np.asarray([metrics.mean_squared_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            cor_test = np.asarray([np.corrcoef(y[test], preds[name]['y_test'][test])[0, 1] for train, test in cv.split(X, y)])

            stats_folds_list.append(
                np.concatenate((np.array([name] * len(mae_test))[:, None],
                                np.array(["CV%i" % cv for cv in range(len(mae_test))])[:, None],
                                mae_test[:, None], r2_test[:, None], mse_test[:, None], cor_test[:, None],
                                np.asarray(preds[name]['time'])[:, None]), axis=1))
            stats_list.append([name, np.mean(mae_test), np.mean(r2_test), np.mean(mse_test), np.mean(cor_test), np.mean(preds[name]['time'])])

        stats_df = pd.DataFrame(stats_list, columns=['model', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time"])

        stats_folds_df = pd.DataFrame(np.concatenate(stats_folds_list, axis=0),
                                      columns=['model', 'fold', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time"])

    model_params = None

    return stats_df, stats_folds_df, model_params
