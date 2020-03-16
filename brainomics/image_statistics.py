#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 11/22/19

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import pandas as pd
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
from collections import OrderedDict

def univ_stats(Y, formula, data):
    """
    Parameters
    ----------
    Y: array (n_subjects, n_features)
    formula: str eg. "age + sex + site"
    data: DataFrame, containing value of formula terms

    """
    X, t_contrasts, f_contrasts = mulm.design_matrix(formula=formula, data=data)
    mod_mulm = mulm.MUOLS(Y, X).fit()
    aov_mulm = OrderedDict((term, mod_mulm.f_test(f_contrasts[term], pval=True)) for term in f_contrasts)

    return mod_mulm, aov_mulm

def plot_univ_stats(univstats, mask_img, data=None, grand_mean=None, pdf_filename=None, thres_nlpval=3, skip_intercept=True):
    """

    Parameters
    ----------
    univstats: dict(indendant_variable:[[fstats], [pvalues]], ...)
    mask_img: mask image, such that sum(mask_img.get_data() > 0) == stat[iv][0].shape[0]
    data: DataFrame (n_variables, n_subjects) used to build the design matrix. Variables are ploted against grand mean
    grand_mean: array (n_subjects, ) subject grand mean
    pdf_filename: str,
    thres_nlpval: Threshold -log pvalue (default 3)

    Returns
    -------

    """
    if pdf_filename:
        pdf = PdfPages(pdf_filename)
    fig_title = os.path.splitext(os.path.basename(pdf_filename))[0]

    if skip_intercept:
        univstats = univstats.copy()
        univstats.pop("Intercept")

    mask_arr = mask_img.get_data() > 0
    ####################################################################################################################
    # Plots

    # Plot brain maps
    fig, ax = plt.subplots(nrows=len(univstats), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(univstats):
        # print(cpt, var)
        #ax_title = "%s: -log p-value (y~%s)" % (var ,"+".join(design_mat.columns))
        ax_title = "%s: -log p-value" % var
        map_arr = np.zeros(mask_arr.shape)
        map_arr[mask_arr] = -np.log10(univstats[var][1])
        map_img = nilearn.image.new_img_like(mask_img, map_arr)
        nilearn.plotting.plot_glass_brain(map_img, colorbar=True, threshold=thres_nlpval, title=ax_title,
                                          figure=fig, axes=ax[cpt])

    if pdf:
        pdf.savefig()
    plt.close(fig)

    # Plot p-value histo
    fig, ax = plt.subplots(nrows=len(univstats), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(univstats):
        # print(cpt, var)
        # ax_title = "%s: histo p-value (y~%s)" % (var, "+".join(design_mat.columns))
        ax_title = "%s: histo p-value" % var
        ax[cpt].hist(univstats[var][1], bins=100)
        ax[cpt].set_title(ax_title)

    plt.tight_layout()

    if pdf_filename:
        pdf.savefig()
    plt.close(fig)

    # data variables vs subject grand mean
    if data is not None and grand_mean is not None:
        df = data.copy()
        df["grand_mean"] = grand_mean
        categoricals = df.select_dtypes(exclude=['int', 'float']).columns
        fig, ax = plt.subplots(nrows=data.shape[1], ncols=1, figsize=(8.27, 11.69))
        fig.suptitle(fig_title)
        for cpt, var in enumerate(data.columns):
            # cpt, var = 2, 'diagnosis'
            print(cpt, var, var in categoricals)
            if (var in categoricals) or (len(df[var].unique()) <= 2):
                # keep only levels with at least 5 samples
                levels_count = df[var].value_counts()
                levels_keep = levels_count[levels_count > 5].index
                df_ = df[df[var].isin(levels_keep)]
                sns.violinplot(x=var, y="grand_mean", data=df_, ax=ax[cpt])
            else:
                sns.scatterplot(x=var, y="grand_mean", data=df, ax=ax[cpt])

        plt.tight_layout()

        if pdf_filename:
            pdf.savefig()
        plt.close(fig)

    if pdf_filename:
        pdf.close()


def residualize(Y, formula_res, data, formula_full=None):
    """
    Residualisation of adjusted residualization.

    Parameters
    ----------
    Y: array (n, p), dependant variables
    formula_res: str, residualisation formula ex: "site":
    1) Fit  Y = b0 + b1 site + eps
    2) Return Y - b0 - b1 site
    data: DataFrame of independant variables
    formula_full:  str, full model formula (default None) ex: "age + sex + site + diagnosis". If not Null residualize
    performs an adjusted residualization:
    1) Fit Y = b1 age + b2 sex + b3 site + b4 diagnosis + eps
    2) Return Y - b3 site

    Returns
    -------
    Y: array (n, p), of residualized dependant variables
    """
    if formula_full is None:
        formula_full = formula_res

    res_terms = mulm.design_matrix(formula=formula_res, data=data)[1].keys()

    X, t_contrasts, f_contrasts = mulm.design_matrix(formula=formula_full, data=data)

    # Fit full model
    mod_mulm = mulm.MUOLS(Y, X).fit()

    # mask of terms in residualize formula within full model
    mask = np.array([cont  for term, cont in t_contrasts.items() if term in res_terms]).sum(axis=0) == 1

    return Y -  np.dot(X[:, mask], mod_mulm.coef[mask, :])


def ml_predictions(X, y, estimators, cv=None, mask_arr=None):
    """

    Parameters
    ----------
    X : ndarray, of shape (n_subjects, n_features) or (n_subjects, 1, image_shape) if mask_arr is provided.
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

    Examples
    --------
    >>> from sklearn.datasets import make_regression, make_classification
    >>> import sklearn.linear_model as lm
    from sklearn.neural_network import MLPClassifier

    X, y = make_classification(n_features=50, n_informative=2, random_state=1, class_sep=0.5)

    estimators = dict(
        lr=lm.LogisticRegressionCV(class_weight='balanced'),
        mlp=MLPClassifier(alpha=1, max_iter=1000))

    stats_df, stats_folds_df, model_params = ml_predictions(X, y, estimators, cv=None, mask_arr=None)

    X, y = make_regression(n_features=50, n_informative=2, random_state=1)
    estimators = dict(lr_inter=lm.RidgeCV(), lr_nointer=lm.RidgeCV(fit_intercept=False))
    stats_df, stats_folds_df, model_params = ml_predictions(X, y, estimators, cv=None, mask_arr=None)
    """
    if mask_arr is not None:
        X = X.squeeze()[:, mask_arr]

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
            if(len(np.unique(y)) == 2): # TODO extends for mulinomial classif
                if hasattr(estimators[name], 'decision_function'):
                    results[name]["score_test"] = estimators[name].decision_function(X_test)
                elif hasattr(estimators[name], 'predict_log_proba'):
                    results[name]["score_test"] = estimators[name].predict_log_proba(X_test)[:, 1]
            try:
                results[name]["coefs"] = estimators[name].coef_
            except:
                pass

            if hasattr(estimators[name], 'alpha_'):
                results[name]["best_param"] = estimators[name].alpha_
            elif hasattr(estimators[name], 'C_'):
                results[name]["best_param"] = estimators[name].C_[0]
            else:
                results[name]["best_param"] = np.NaN

        return results

    parallel = Parallel(n_jobs=5)
    cv_ret = parallel(
        delayed(_split_fit_predict)(
            {name:clone(estimators[name]) for name in estimators}, X, y, train, test)
        for train, test in cv.split(X, y))

    # Aggregate predictions
    preds = {name:dict(y_test=np.zeros(len(y)), score_test=np.zeros(len(y)), time=[], coefs=[], best_param=[]) for name in estimators}
    for i, (train, test) in enumerate(cv.split(X, y)):
        for name in estimators:
            preds[name]['y_test'][test] = cv_ret[i][name]['y_test']
            try:
                preds[name]['score_test'][test] = cv_ret[i][name]['score_test']
            except:
                if 'score_test' in preds[name]: preds[name].pop('score_test')
            preds[name]['time'].append(cv_ret[i][name]['time'])
            preds[name]['best_param'].append(cv_ret[i][name]['best_param'])
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
                size_test = np.asarray([len(y[test]) for train, test in cv.split(X, y)])

                folds = pd.DataFrame()
                folds['model'] = [name] * len(baccs_test)
                folds['fold'] = ["CV%i" % fold for fold in range(len(baccs_test))]
                folds['bacc_test'] = baccs_test
                folds['auc_test'] = aucs_test
                for i, lev in enumerate(np.unique(y)):
                    folds['recall%i_test' % lev] = recalls_test[:, i]
                folds["time"] = preds[name]['time']
                folds["best_param"] = preds[name]['best_param']
                folds["size_test"] = size_test
                stats_folds_list.append(folds)
                stats_list.append([name, np.mean(baccs_test), np.mean(aucs_test)] + \
                                      np.asarray(recalls_test).mean(axis=0).tolist() + [np.mean(preds[name]['time']), str(preds[name]['best_param'])])
        stats_folds_df = pd.concat(stats_folds_list)
        stats_df = pd.DataFrame(stats_list, columns=['model', 'bacc_test', 'auc_test'] + \
                                                    ['recall%i_test' % lev for lev in np.unique(y)] + ["time", "best_param"])
        stats_df["size"] = str(["%i:%i" % (lab, np.sum(y==lab)) for lab in np.unique(y)])

    else:
        for name in preds:
            mae_test = np.asarray([metrics.mean_absolute_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            r2_test = np.asarray([metrics.r2_score(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            mse_test = np.asarray([metrics.mean_squared_error(y[test], preds[name]['y_test'][test]) for train, test in cv.split(X, y)])
            cor_test = np.asarray([np.corrcoef(y[test], preds[name]['y_test'][test])[0, 1] for train, test in cv.split(X, y)])
            size_test = np.asarray([len(y[test]) for train, test in cv.split(X, y)])

            stats_folds_list.append(
                np.concatenate((np.array([name] * len(mae_test))[:, None],
                                np.array(["CV%i" % cv for cv in range(len(mae_test))])[:, None],
                                mae_test[:, None], r2_test[:, None], mse_test[:, None], cor_test[:, None],
                                np.asarray(preds[name]['time'])[:, None],
                                np.asarray(preds[name]['best_param'])[:, None],
                                size_test[:, None]), axis=1))
            stats_list.append([name, np.mean(mae_test), np.mean(r2_test), np.mean(mse_test), np.mean(cor_test),
                               np.mean(preds[name]['time']), str(preds[name]['best_param'])])

        stats_df = pd.DataFrame(stats_list, columns=['model', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time", "best_param"])
        stats_df["size"] = len(y)

        stats_folds_df = pd.DataFrame(np.concatenate(stats_folds_list, axis=0),
                                      columns=['model', 'fold', 'mae_test', 'r2_test', 'mse_test', 'cor_test', "time", "best_param", "size_test"])

    model_params = None

    return stats_df, stats_folds_df, model_params

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from brainomics.image_statistics import univ_stats, plot_univ_stats, ml_predictions, residualize

    ####################################################################################################################
    # Residualization
    import seaborn as sns
    # %matplotlib qt

    # Dataset no site effect in age

    age = np.random.uniform(10, 40, size=100)
    sex = np.random.choice([0, 1], 100)
    sex_c = ["X%i" % i for i in sex]
    site = np.array([-1] * 50 + [1] * 50)
    # age = age + 1 * site
    site_c = ["S%i" % i for i in site]

    y0 = -0.1 * age + 0.0 * sex + site + np.random.normal(size=100)

    data = pd.DataFrame(dict(age=age, sex=sex_c, site=site_c, y0=y0))
    Y = np.asarray(data[["y0"]])

    Yres = residualize(Y, formula_res="site", data=data)
    Yadj = residualize(Y, formula_res="site", data=data, formula_full="age + sex + site")

    data["y0res"] = Yres[:, 0]
    data["y0adj"] = Yadj[:, 0]

    # Simple residualization or adjusted residualization works the same
    sns.lmplot("age", "y0", hue="site", data=data)
    sns.lmplot("age", "y0res", hue="site", data=data)
    sns.lmplot("age", "y0adj", hue="site", data=data)

    # Dataset with site effect in age

    age = np.random.uniform(10, 40, size=100)
    sex = np.random.choice([0, 1], 100)
    sex_c = ["X%i" % i for i in sex]
    site = np.array([-1] * 50 + [1] * 50)
    age = age + 5 * site
    site_c = ["S%i" % i for i in site]

    y0 = -0.1 * age + 0.0 * sex + site + np.random.normal(size=100)

    data = pd.DataFrame(dict(age=age, sex=sex_c, site=site_c, y0=y0))
    Y = np.asarray(data[["y0"]])

    Yres = residualize(Y, formula_res="site", data=data)
    Yadj = residualize(Y, formula_res="site", data=data, formula_full="age + sex + site")

    data["y0res"] = Yres[:, 0]
    data["y0adj"] = Yadj[:, 0]

    # Requires adjusted residualization
    sns.lmplot("age", "y0", hue="site", data=data)
    sns.lmplot("age", "y0res", hue="site", data=data)
    sns.lmplot("age", "y0adj", hue="site", data=data)
