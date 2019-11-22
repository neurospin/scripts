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

def univariate_statistics(NI_arr, ref_img, design_mat, fig_title=None, pdf=None, mask_arr=None, thres_nlpval=3):
    """
    Perform univariate statistics

    Parameters
    ----------
    NI_arr :  ndarray, of shape (n_subjects, 1, image_shape).
    ref_img : image
    design_mat : DataFrame of the design matrix
    fig_title : str figure title
    pdf : PdfPages(pdf_filename), optional
    mask_arr : ndarray of image_shape, if None use `preproc.compute_brain_mask(NI_arr, ref_img).get_data() > 0`
    thres_nlpval : float, threshold of neg log p-values

    Returns
    -------
        dict : of variables, where each variable is a dict of 'stat' and 'pval' computed on `mask_arr`.
            Use as `NI_arr[:, :, mask_arr].squeeze()`
        mask_arr : ndarray, the mask
    """

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
        print(var)
        con = np.zeros(Xdf_dummy.shape[1])
        con[variables[var]['idx']] = 1
        variables[var]['contrast'] = con

    for var in Xdf_cat.columns:
        print(var)
        con = np.zeros((Xdf_dummy.shape[1], Xdf_dummy.shape[1]))
        indices = np.arange(variables[var]['idx'], variables[var]['idx'] + variables[var]['len'])
        con[indices, indices] = 1
        variables[var]['contrast'] = con

    X = np.asarray(Xdf_dummy)
    mod = mulm.MUOLS(Y, X)
    mod.fit()

    for var in Xdf_num.columns:
        print(var)
        tvals, pvals, dof = mod.t_test(variables[var]['contrast'], pval=True, two_tailed=True)
        variables[var]['stat'] = tvals.squeeze()
        variables[var]['pval'] = pvals.squeeze()

    for var in Xdf_cat.columns:
        print(var)
        fvals, pvals = mod.f_test(variables[var]['contrast'], pval=True)
        variables[var]['stat'] = fvals.squeeze()
        variables[var]['pval'] = pvals.squeeze()

    ####################################################################################################################
    # Plots

    # Plot brain maps
    fig, ax = plt.subplots(nrows=len(variables), ncols=1, figsize=(8.27, 11.69))
    fig.suptitle(fig_title)
    for cpt, var in enumerate(variables):
        print(cpt, var)
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
        print(cpt, var)
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
        print(cpt, var)
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

    # Clean variables
    for var in variables:
        variables[var] = dict(stat=variables[var]['stat'], pval=variables[var]['pval'])

    return variables, mask_arr
