#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 16:28:43 2021

@author: ed203246
"""
import sys
import os
import time

import numpy as np
import nibabel
import pandas as pd
import matplotlib.pylab as plt
import nilearn
from nilearn import plotting
from nilearn.image import resample_to_img
from sklearn.decomposition import PCA
import argparse
import glob
import itertools

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Stats
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Compute Xk
from parsimony.decomposition import PCAL1L2TV
import parsimony.decomposition.pca_tv as pca_tv

# use it to compute wmh deflated mean
pca_tv = PCAL1L2TV(n_components=3, l1=0.1, l2=0.1, ltv=0.1, Atv=None)

################################################################################
# Config

sys.path.append('/home/ed203246/git/scripts/2021_wmh_memento+rundmc')
from file_utils import load_npy_nii

#import nilearn.datasets
#import brainomics.image_resample

FS = "/home/ed203246/data"

#%% MEMENTO_RUNDMC
MEMENTO_RUNDMC_PATH = "{FS}/2021_wmh_memento+rundmc".format(FS=FS)
MEMENTO_RUNDMC_DATA = os.path.join(MEMENTO_RUNDMC_PATH, "data")

#%% MEMENTO
MEMENTO_PATH = "{FS}/2017_memento/analysis/WMH".format(FS=FS)
MEMENTO_DATA = os.path.join(MEMENTO_PATH, "data")
MEMENTO_MODEL = os.path.join(MEMENTO_PATH, "models/pca_enettv_0.000010_1.000_0.001")
os.listdir(MEMENTO_MODEL)

#%% RUNDMC
RUNDMC_PATH = "{FS}/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca".format(FS=FS)
RUNDMC_DATA = os.path.join(RUNDMC_PATH, "data")
RUNDMC_MODEL = os.path.join(RUNDMC_PATH, "models/pca_enettv_0.000035_1.000_0.005")

os.listdir(RUNDMC_MODEL)

################################################################################
# Utils
def flat_to_img(mask_img, flat_values):
    val_arr = np.zeros(mask_img.get_fdata().shape)
    val_arr[mask_img.get_fdata() != 0] = flat_values.squeeze()
    return nilearn.image.new_img_like(mask_img, val_arr)


def ttest(formula, hypothesis, data):
    dv, design = [x.strip() for x in formula.split("~")]
    contrast_ = hypothesis.split("=")[0].strip() if "=" in hypothesis else hypothesis.strip()

    lm = smf.ols(formula, data=phenotypes).fit()
    # lm.summary()
    ttest_ = lm.t_test(hypothesis)

    return pd.DataFrame([[formula, dv, design, contrast_,
                   ttest_.tvalue[0, 0], float(ttest_.pvalue), ttest_.effect[0],
                   ttest_.conf_int()[0, 0], ttest_.conf_int()[0, 1]]],
                 columns=['formula', 'var', 'design', 'test', 'tstat', 'pval', 'coef',
                          'ci2.5%', 'ci97.5%'])

################################################################################
# Load datasets

def load_data(model_filename, loadings_img_filename, mask_img_filename,
              wmh_filename, data_shape, participants_with_phenotypes_filename,
              tissue_filename):
    # Phenoypes
    phenotypes = pd.read_csv(participants_with_phenotypes_filename)

    # Model
    model = np.load(model_filename)
    U, d, V, PC, explained_variance = model['U'], model['d'], model['V'], model['PC'], \
                                      model['explained_variance']

    assert phenotypes.shape[0] == PC.shape[0]
    phenotypes["PC1"] = PC[:, 0]
    phenotypes["PC2"] = PC[:, 1]
    phenotypes["PC3"] = PC[:, 2]

    # Loading_img
    loadings_img = nibabel.load(loadings_img_filename)

    # mask_img
    mask_img = nibabel.load(mask_img_filename)
    assert loadings_img.get_fdata().shape[:-1] == mask_img.get_fdata().shape
    assert np.all(loadings_img.affine == mask_img.affine)

    mask_arr = mask_img.get_fdata() == 1
    assert mask_arr.sum() == data_shape[1]

    # Data
    data = load_npy_nii(wmh_filename)
    if isinstance(data, np.ndarray):
        pass
    elif hasattr(data, 'affine'):
        data_img = data
        data = data_img.get_fdata()[mask_arr].T

    phenotypes["wmh_tot"] = data.sum(axis=1)

    assert mask_arr.sum() == data.shape[1] == data_shape[1]
    assert data.shape[0] == data_shape[0]

    # wmh_by_tissue
    tissue_img = nibabel.load(tissue_filename)
    tissue_img = resample_to_img(source_img=tissue_img, target_img=mask_img, interpolation='nearest')
    tissue_arr = tissue_img.get_fdata().astype(int)
    tissue_labels = dict(cortex=1, ventricles=2, deepwm=3)

    # Tissue map
    # tissue, lab = "cortex", 1
    # tissue_img.to_filename("/tmp/tissue_rundmc.nii.gz")
    # arr_3d = np.zeros(mask_arr.shape)
    # arr_3d[(tissue_arr == lab) & mask_arr] = data[:, tissue_arr[mask_arr] == lab].mean(axis=0)
    # nilearn.image.new_img_like(mask_img, arr_3d).to_filename("/tmp/wmh_cortex_rundmc.nii.gz")

    assert np.all(tissue_arr == tissue_img.get_fdata())
    wmh_pc_by_tissue = pd.DataFrame({"wmh_pc_%s" % tissue:PCA(n_components=1).fit_transform(data[:, tissue_arr[mask_arr] == lab]).ravel()
        for tissue, lab in tissue_labels.items()})
    wmh_sum_by_tissue = pd.DataFrame({"wmh_sum_%s" % tissue:data[:, tissue_arr[mask_arr] == lab].sum(axis=1)
        for tissue, lab in tissue_labels.items()})
    phenotypes = pd.concat([phenotypes, wmh_pc_by_tissue, wmh_sum_by_tissue], axis=1)

    # center data
    data = data - data.mean(axis=0)

    # Compute mean deflated data
    pca_tv.V = V
    X = data
    n_components = 3
    rsquared = np.zeros((n_components))
    X_deflated_mean = np.zeros((n_components, X.shape[1]))
    X_deflated_std = np.zeros((n_components, X.shape[1]))

    for j in range(1, n_components + 1):
        # model.n_components = j + 1
        X_predict = pca_tv.predict(X, n_components=j)
        sse = np.sum((X - X_predict) ** 2)
        ssX = np.sum(X ** 2)
        rsquared[j - 1] = 1 - sse / ssX
        X_deflated_mean[j - 1, :] = X_predict.mean(axis=0)
        X_deflated_std[j - 1, :] = X_predict.std(axis=0)

    # Check we recover the explained_variance
    np.allclose(explained_variance, rsquared)
    wmh_deflated_mean_img = nilearn.image.concat_imgs([flat_to_img(mask_img, X_deflated_mean[i, :]) for i in range(X_deflated_mean.shape[0])])
    wmh_deflated_std_img = nilearn.image.concat_imgs([flat_to_img(mask_img, X_deflated_std[i, :]) for i in range(X_deflated_std.shape[0])])

    # Correlation between WMH and PCs: wmh_pc_cor_map
    from sklearn.preprocessing import StandardScaler
    data_s = StandardScaler().fit_transform(data)
    PC_s = StandardScaler().fit_transform(PC)
    wmh_pc_cor_flat = np.dot(data_s.T,  PC_s) / data_s.shape[0]
    # pd.Series(wmh_pc_cor_flat[:, 2]).describe()
    del data_s, PC_s
    wmh_pc_cor_img = nilearn.image.concat_imgs([flat_to_img(mask_img, wmh_pc_cor_flat[:, i]) for i in range(wmh_pc_cor_flat.shape[1])])

    # Brain Parenchymal Fraction
    phenotypes['bpf'] = phenotypes['brain_volume'] / phenotypes['tiv']
    return model, loadings_img, mask_img, tissue_img, data, wmh_pc_cor_img, wmh_deflated_mean_img, wmh_deflated_std_img, phenotypes

################################################################################
# statistics

def do_statistics(phenotypes, model, stats_filename):

    clinic_vars = ['MMSE', 'processing_speed', 'executive_functions', 'memory']
    demo_vars = ['age', 'sex', 'education']
    mri_vars = ['lacune_nb', 'mb_nb', 'bpf', 'tiv', 'wmh_tot', 'wmh_sum_cortex', 'wmh_sum_ventricles', 'wmh_sum_deepwm']

    ############################################################################
    # 3) Exploratory analysis and plots
    # 3.1) Correlation between variables
    fig_filename = os.path.splitext(stats_filename)[0] + '.pdf'
    pdf_device = PdfPages(fig_filename)

    df = phenotypes[clinic_vars + demo_vars + mri_vars + ['PC1', 'PC2']]
    df = df[[v for v in df.columns if df[v].var() != 0]]
    R_ = df.corr()
    R_pval = R_.copy(); R_pval.iloc[:, :] = np.NAN
    R = R_.copy(); R.iloc[:, :] = np.NAN
    for v1, v2 in itertools.product(df.columns, df.columns):
        print(v1, v2)
        df_ = df[[v1, v2]].dropna()
        R.loc[v1, v2], R_pval.loc[v1, v2] = scipy.stats.pearsonr(df_[v1], df_[v2]) if v1 != v2 else (1, 0)

    assert np.allclose(R_, R)
    # Bonferoni pvalues correction
    R_pval_fwer = R_pval * R.shape[0] * (R.shape[0] - 1) / 2
    R_pval_fwer[R_pval_fwer > 1] = 1
    # R_[R_pval_fwer > 0.05] = np.nan

    # Threshold on pvalues
    R_[R_pval > 0.05] = np.nan

    plt.figure(); plt.clf()
    fig_cor = sm.graphics.plot_corr(R_, xnames=R.index, title="Correlation matrix, pval(fwer) < 0.05")
    pdf_device.savefig()
    #plt.savefig()
    #plt.close()

    # PC1 x PC2
    plt.figure(); plt.clf()
    pc1_var, pc2_var = model['explained_variance'][0], model['explained_variance'][1] - model['explained_variance'][0]
    ax = sns.regplot(x='PC1', y='PC2', data=phenotypes)
    ax.set_xlabel('PC1 (%.2f%% of variance)' % (pc1_var * 100))
    ax.set_ylabel('PC2 (%.2f%% of variance)' % (pc2_var * 100))
    pdf_device.savefig()

    """
    sns.relplot(x='PC1', y='PC2', hue='MMSE', data=phenotypes, palette=sns.color_palette("Spectral", as_cmap=True), alpha=0.5)
    from matplotlib import cm
    from sklearn.svm import SVR
    from sklearn.svm import OneClassSVM

    df_ = phenotypes[['PC1', 'PC2', "MMSE"]].dropna()
    X = df_[['PC1', 'PC2']].values
    y = df_["MMSE"].values
    #not_outliers = OneClassSVM(gamma=1/100).fit_predict(X)
    not_outliers = (X[:, 0] < 0.06) & (X[:, 1] < 0.06) # MEMENTO
    not_outliers = np.ones(df_.shape[0])
    not_outliers = (X[:, 0] < 0.06) & (X[:, 1] < 0.06) # RUNDMC

    sns.relplot(x='PC1', y='PC2', hue='MMSE', data=df_[not_outliers == 1], palette=sns.color_palette("Spectral", as_cmap=True), alpha=0.5)
    #sns.relplot(x='PC1', y='PC2', hue='MMSE', data=df_, palette=sns.color_palette("Spectral", as_cmap=True), alpha=0.5)

    X, y = X[not_outliers == 1, :], y[not_outliers == 1]

    svr = SVR(gamma=100000).fit(X, y)
    svr = SVR(gamma=1000).fit(X, y)

    # svr = SVR(gamma=100).fit(X, y) # MEMENTO

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1e-5, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp = GaussianProcessRegressor(kernel=1 * RBF(1e-3, (1e-2, 1e2)))

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)

    pc1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    pc2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    pc1_grid, pc2_grid = np.meshgrid(pc1_range, pc2_range)
    xy_grid = np.vstack([pc1_grid.ravel(), pc2_grid.ravel()]).T
    # z_predict = svr.predict(np.vstack([pc1_grid.ravel(), pc2_grid.ravel()]).T).reshape(pc1_grid.shape)
    z_predict, z_sigma = gp.predict(xy_grid, return_std=True)
    # z_predict = z_predict * np.exp(- 1/ 2 * z_sigma)
    z_predict = z_predict.reshape(pc1_grid.shape)

    plt.figure()
    plt.clf()
    im = plt.imshow(z_predict[::-1, :], cmap=cm.coolwarm)  # drawing the function
    # adding the Contour lines with labels
    cset = plt.contour(z_predict[::-1, :], 3, linewidths=2, cmap=cm.gray)
    plt.clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    plt.colorbar(im)  # adding the colobar on the right
    plt.show()

    # Plot the surface.
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(pc1_grid, pc2_grid, z_predict, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("MMSE")
    plt.show()
    ############################################################################
    """
    # PC1 x PC2 x MMSE
    plt.figure(); plt.clf()
    sns.pairplot(phenotypes[['PC1', 'PC2', 'MMSE']])
    pdf_device.savefig()

    plt.clf()

    ################################################################################
    #     1) links between component values and mri markers (rough data, no adjustment)
    #     • c1 vs wmh volume (ed)
    #     • c1 vs subcortical wmh volume (ed masks to be done<1cm from cortex)
    #     • c1 vs periventricular wmh volume (ed masks to be done < 1cm ventricules)
    #     • c1 vs deep wmh volume (the rest)
    #     • c1 vs lacune number (eric process memento and anil send)
    #     • c1 vs mb number (eric anil process )
    # same for pc2

    stats_pc_mri_markers = pd.concat(
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
            for design in ['PC1'] for con in ['PC1=0'] for v in mri_vars] +
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
         for design in ['PC2'] for con in ['PC2=0'] for v in mri_vars] +
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
         for design in ['PC1 + PC2'] for con in ['PC1=0', 'PC2=0', 'PC1-PC2=0'] for v in mri_vars]
    )

    stats_pc_demo = pd.concat([
        ttest(formula='PC1 ~ age + sex + education', hypothesis='age=0', data=phenotypes),
        ttest(formula='PC1 ~ age + sex + education', hypothesis='sex=0', data=phenotypes),
        ttest(formula='PC1 ~ age + sex + education', hypothesis='education=0', data=phenotypes),
        ttest(formula='PC2 ~ age + sex + education', hypothesis='age=0', data=phenotypes),
        ttest(formula='PC2 ~ age + sex + education', hypothesis='sex=0', data=phenotypes),
        ttest(formula='PC2 ~ age + sex + education', hypothesis='education=0', data=phenotypes)
    ])

    ################################################################################
    #     2) links between component values and clinical status (systematic adjustment for age and level of education) -> if difficulties with age, this should be extremely clearly explained+++
    #     • c1 vs mmse
    #     • c1 vs processing speed (tmta memento, compound score rundmc)
    #     • c1 vs executive functions (fluencies memento eric check, ok rundmc)
    #     • c1 vs memory (fcsrt memento, compound rundmc)
    # Same for C2
    # model:
    # mmse ~ pc1 + age + sex + education + brain atrophy + lacune nb + mb

    # Clinic vs PCs: Original design
    stats_pc_clinic_orig = pd.concat(
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
            for design in ['PC1 + age + sex + education + lacune_nb + mb_nb + bpf']
            for con in ['PC1=0']
            for v in clinic_vars] +
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
            for design in ['PC2 + age + sex + education + lacune_nb + mb_nb + bpf']
            for con in ['PC2=0']
            for v in clinic_vars] +
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
            for design in ['PC1 + PC2 + age + sex + education + lacune_nb + mb_nb + bpf']
            for con in ['PC1=0', 'PC2=0', 'PC1-PC2=0']
            for v in clinic_vars]
    )

    # Clinic vs PCs: Since age and bpf are correlated with PC1, ommit age and bpf, but correct for tiv
    designs = ['PC1 + PC2 + sex + education + lacune_nb + mb_nb + tiv']
    #vars = ['MMSE', 'processing_speed', 'executive_functions', 'memory']
    contrasts = ['PC1=0', 'PC2=0', 'PC1-PC2=0']
    stats_pc_clinic = pd.concat(
        [ttest(formula='%s ~ %s' % (v, design), hypothesis=con, data=phenotypes)
            for design in designs for con in contrasts for v in clinic_vars])

    ####################################################################################################################
    # Partial Regression Plots & Component plus Residual(CCPR) Plots
    # https://www.statsmodels.org/stable/examples/notebooks/generated/regression_plots.html
    for v in clinic_vars:
        formula = '%s ~ %s' % (v, designs[0])
        lm = smf.ols(formula, data=phenotypes).fit()

        # = Print ols.summary() into pdf =
        # formula = 'MMSE ~ PC1 + PC2 + sex + education + lacune_nb + mb_nb + tiv'
        # lm = smf.ols(formula, data=phenotypes).fit()
        txt = lm.summary()
        textfig = plt.figure(figsize=(11.69, 8.27))
        textfig.clf()
        # The default transform specifies that text is in data coords, alternatively, you can specify text in axis coords
        # (0,0 is lower-left and 1,1 is upper-right)
        textfig.text(0.1, 0.1, txt, transform=textfig.transFigure, size=10, fontfamily='monospace')
        pdf_device.savefig()
        # = END =

        # Partial Regression Plots
        plt.clf()
        fig = sm.graphics.plot_partregress_grid(lm, exog_idx=['PC1', 'PC2'])
        fig.tight_layout(pad=1.0)
        pdf_device.savefig()

        # Component - Component plus Residual(CCPR) Plots
        # plt.clf()
        # fig = sm.graphics.plot_ccpr_grid(lm, exog_idx=['PC1', 'PC2'])
        # fig.tight_layout(pad=1.0)
        # pdf_device.savefig()
    #
    pdf_device.close()

    with pd.ExcelWriter(stats_filename) as writer:
        R.to_excel(writer, sheet_name='corr')
        R_pval_fwer.to_excel(writer, sheet_name='pval_fwer')
        stats_pc_demo.to_excel(writer, sheet_name='stats_pc_demo', index=False)
        stats_pc_mri_markers.to_excel(writer, sheet_name='stats_pc_mri_markers', index=False)
        stats_pc_clinic.to_excel(writer, sheet_name='stats_pc_clinic', index=False)
        stats_pc_clinic_orig.to_excel(writer, sheet_name='stats_pc_clinic_orig', index=False)

    print(stats_filename)

# tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_cortex_ventricles_dill5mm_deepwm_mni_1mm.nii.gz")
# tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_cortex_ventricles_dill10mm_deepwm_mni_1mm.nii.gz")
tissue_filename = os.path.join(MEMENTO_RUNDMC_DATA, "mask_other_ventricles_dill10mm_jhu-icbm-dti-81_mni_1mm.nii.gz")

################################################################################
# MEMENTO
# /home/ed203246/data/2017_memento/analysis/WMH/models/pca_enettv_0.000010_1.000_0.001

if False:
    model_filename = os.path.join(MEMENTO_MODEL, "model.npz")
    loadings_img_filename = os.path.join(MEMENTO_MODEL, "components-brain-maps.nii.gz")
    mask_img_filename = os.path.join(MEMENTO_DATA, "mask.nii.gz")
    wmh_filename = os.path.join(MEMENTO_DATA, "WMH_arr_msk.npy")
    data_shape = (1755, 116037)
    # phenotypes_filename = os.path.join(MEMENTO_PATH , "population.csv")
    # participants_with_phenotypes_filename = ??
    components_wmh_cor_img_filename = loadings_img_filename.replace("components-brain-maps", "components-wmh-cor-brain-maps")
    wmh_deflated_mean_img_filename = loadings_img_filename.replace("components-brain-maps", "wmh-deflated-mean-maps")
    wmh_deflated_std_img_filename = loadings_img_filename.replace("components-brain-maps", "wmh-deflated-std-maps")
    participants_with_phenotypes_filename = os.path.join(MEMENTO_PATH , "population_with_phenotypes.csv")

    data_filename = os.path.join(MEMENTO_MODEL, "pcs-mrimarkers-demo-clinic_pcatv_202103.xlsx")
    stats_filename = os.path.join(MEMENTO_MODEL, "stats_pcatv_202103.xlsx")

    model, loadings_img, mask_img, tissue_img, data, wmh_pc_cor_img, wmh_deflated_mean_img, wmh_deflated_std_img, phenotypes = \
        load_data(model_filename, loadings_img_filename, mask_img_filename,
                  wmh_filename, data_shape, participants_with_phenotypes_filename,
                  tissue_filename)

    # lacune_nb & mb_nb are missing
    phenotypes["lacune_nb"] = 0
    phenotypes["mb_nb"] = 0
    phenotypes["memory"] = 0

    wmh_pc_cor_img.to_filename(components_wmh_cor_img_filename)
    wmh_deflated_mean_img.to_filename(wmh_deflated_mean_img_filename)
    wmh_deflated_std_img.to_filename(wmh_deflated_std_img_filename)
    phenotypes.to_excel(data_filename)

    do_statistics(phenotypes, model, stats_filename)
    print(stats_filename)

################################################################################
# RUNDMC
# /home/ed203246/data/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000035_1.000_0.005

if False:
    model_filename = os.path.join(RUNDMC_MODEL, "model.npz")
    loadings_img_filename = os.path.join(RUNDMC_MODEL, "components-brain-maps.nii.gz")
    mask_img_filename = os.path.join(RUNDMC_DATA, "mask.nii.gz")
    # See /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py
    wmh_filename = os.path.join(RUNDMC_DATA, "WMH_2006.nii.gz")
    #participants_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants.csv")
    participants_with_phenotypes_filename = os.path.join(RUNDMC_DATA, "WMH_2006_participants_with_phenotypes.csv")
    data_shape = (267, 371278)

    components_wmh_cor_img_filename = loadings_img_filename.replace("components-brain-maps", "components-wmh-cor-brain-maps")
    wmh_deflated_mean_img_filename = loadings_img_filename.replace("components-brain-maps", "wmh-deflated-mean-maps")
    wmh_deflated_std_img_filename = loadings_img_filename.replace("components-brain-maps", "wmh-deflated-std-maps")

    data_filename = os.path.join(RUNDMC_MODEL, "pcs-mrimarkers-demo-clinic_pcatv_202103.xlsx")
    stats_filename = os.path.join(RUNDMC_MODEL, "stats_pcatv_202103.xlsx")

    model, loadings_img, mask_img, tissue_img, data, wmh_pc_cor_img, wmh_deflated_mean_img, wmh_deflated_std_img, phenotypes = \
        load_data(model_filename, loadings_img_filename, mask_img_filename,
                  wmh_filename, data_shape, participants_with_phenotypes_filename,
                  tissue_filename)

    wmh_pc_cor_img.to_filename(components_wmh_cor_img_filename)
    wmh_deflated_mean_img.to_filename(wmh_deflated_mean_img_filename)
    wmh_deflated_std_img.to_filename(wmh_deflated_std_img_filename)
    phenotypes.to_excel(data_filename)

    do_statistics(phenotypes, model, stats_filename)
    print(stats_filename)

################################################################################
"""
PATHS

MEMENTO:
/home/ed203246/data/2017_memento/analysis/WMH/models/pca_enettv_0.000010_1.000_0.001/

RUNDMC
/home/ed203246/data/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000035_1.000_0.005/wmh-deflated-std-maps.nii.gz
/home/ed203246/data/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000035_1.000_0.005/stats_pcatv_202103.xlsx

cd /home/ed203246/data/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/models/pca_enettv_0.000035_1.000_0.005/
fsleyes components-brain-maps.nii.gz wmh-deflated-std-maps.nii.gz
"""
