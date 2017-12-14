#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:42:03 2017

@author: ad247405
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import nibabel as nib
import json
from nilearn import plotting
from nilearn import image
from scipy.stats.stats import pearsonr
import shutil
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import parsimony.utils.check_arrays as check_arrays
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns


WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/data"

U_vip = np.load(os.path.join(WD,"U_vip.npy"))
U_vip_scz = np.load(os.path.join(WD,"U_vip_scz.npy"))
U_vip_con = np.load(os.path.join(WD,"U_vip_con.npy"))

y_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/y.npy")
X_vip = np.load("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/data/data_by_site/VIP/X.npy")

pop_vip = pd.read_csv("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/population_and_scores.csv")
age = pop_vip["age"].values
panss_neg = pop_vip["PANSS_NEGATIVE"]
panss_pos = pop_vip["PANSS_POSITIVE"]
panss_galp = pop_vip["PANSS_GALPSYCHOPAT"]
panss_comp = pop_vip["PANSS_COMPOSITE"]
fast = pop_vip["FAST_TOT"]
hallu_panssP3 = pop_vip["PANSS_P3"]
dose = np.load("/neurospin/brainomics/2016_schizConnect/analysis/VIP/VBM/data/treatment/dose_ongoing_treatment.npy")

panss_pos_scz = panss_pos[y_vip==1]
panss_neg_scz = panss_neg[y_vip==1]
panss_comp_scz = panss_comp[y_vip==1]
hallu_panssP3_scz  = hallu_panssP3[y_vip==1]
fast_scz = fast[y_vip==1]



plt.plot(panss_pos_scz,panss_neg_scz,'o')
plt.xlabel("PANSS positive")
plt.ylabel("PANSS negative")
plt.savefig("/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/\
VBM/all_subjects/results/pcatv_scz/results/projection_vip/panss.png")

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/all_subjects/results/pcatv_scz/results/projection_vip/TB_PSY_VIE_HALLU"
TB_PSY_VIE_HALLU = pop_vip["TB_PSY_VIE_HALLU"].values[y_vip==1]
for i in range(10):
    plt.figure()
    df = pd.DataFrame()
    df["TB_PSY_VIE_HALLU"] = TB_PSY_VIE_HALLU[np.array(np.isnan(TB_PSY_VIE_HALLU)==False)]
    df["score"] = U_vip_scz[np.array(np.isnan(TB_PSY_VIE_HALLU)==False),i]
    T,pvalue = scipy.stats.ttest_ind(df["score"].values[df["TB_PSY_VIE_HALLU"].values==1],\
                                     df["score"].values[df["TB_PSY_VIE_HALLU"].values==0])
    sns.set_style("whitegrid")
    sns.set(font_scale=1.3)
    ax = sns.violinplot(x="TB_PSY_VIE_HALLU", y="score", hue="TB_PSY_VIE_HALLU", data=df,linewidth = 3)
    plt.tight_layout()
    plt.legend(loc='lower center',ncol=2)
    plt.ylabel("Score on component %r"%i)
    plt.title(("T : %s and pvalue = %r"%(np.around(T,decimals=3),pvalue)))
    plt.savefig(os.path.join(output,"comp%s"%((i+1))))



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/panss_pos"
for i in range(10):
    df = pd.DataFrame()
    df["panss_pos"] = panss_pos_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_pos +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"PANSS POSITIVE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["panss_pos"],mod.pvalues["panss_pos"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/panss_neg"
for i in range(10):
    df = pd.DataFrame()
    df["panss_neg"] = panss_neg_scz[np.array(np.isnan(panss_neg_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_neg_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_neg_scz)==False)]
    mod = ols("U ~ panss_neg +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"PANSS NEGATIVE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["panss_neg"],mod.pvalues["panss_neg"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/panss_comp"
for i in range(10):
    df = pd.DataFrame()
    df["panss_comp"] = panss_comp_scz[np.array(np.isnan(panss_comp_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_comp_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_comp_scz)==False)]
    mod = ols("U ~ panss_comp +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"PANSS COMPOSITE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["panss_comp"],mod.pvalues["panss_comp"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/fast"
for i in range(10):
    df = pd.DataFrame()
    df["fast"] = fast_scz[np.array(np.isnan(fast_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(fast_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(fast_scz)==False)]
    mod = ols("U ~ fast +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"FAST effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["fast"],mod.pvalues["fast"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/dose"
for i in range(10):
    df = pd.DataFrame()
    df["dose"] = dose[np.array(np.isnan(dose)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(dose)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(dose)==False)]
    mod = ols("U ~ dose +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"dose effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["dose"],mod.pvalues["dose"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/hallu_panssp3"
for i in range(10):
    df = pd.DataFrame()
    df["hallu_panssP3_scz"] = hallu_panssP3_scz[np.array(np.isnan(hallu_panssP3_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(hallu_panssP3_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(hallu_panssP3_scz)==False)]
    mod = ols("U ~ hallu_panssP3_scz +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"hallu_panssP3_scz effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["hallu_panssP3_scz"],mod.pvalues["hallu_panssP3_scz"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/CVLT_RILT_RC_tot"
CVLT_RILT_RC_tot = pop_vip["CVLT_RILT_RC_tot"].values[y_vip==1]

for i in range(10):
    df = pd.DataFrame()
    df["CVLT_RILT_RC_tot"] = CVLT_RILT_RC_tot[np.array(np.isnan(CVLT_RILT_RC_tot)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(CVLT_RILT_RC_tot)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(CVLT_RILT_RC_tot)==False)]
    mod = ols("U ~ CVLT_RILT_RC_tot +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"CVLT_RILT_RC_tot effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["CVLT_RILT_RC_tot"],mod.pvalues["CVLT_RILT_RC_tot"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/WAIS_COMPL_IM_STD"
WAIS_COMPL_IM_STD = pop_vip["WAIS_COMPL_IM_STD"].values[y_vip==1]
for i in range(10):
    df = pd.DataFrame()
    df["WAIS_COMPL_IM_STD"] = WAIS_COMPL_IM_STD[np.array(np.isnan(WAIS_COMPL_IM_STD)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(WAIS_COMPL_IM_STD)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(WAIS_COMPL_IM_STD)==False)]
    mod = ols("U ~ WAIS_COMPL_IM_STD +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"WAIS Picture Completion \n score effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["WAIS_COMPL_IM_STD"],mod.pvalues["WAIS_COMPL_IM_STD"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))

output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/WAIS_COD_STD"
WAIS_COD_STD = pop_vip["WAIS_COD_STD"].values[y_vip==1]
for i in range(10):
    df = pd.DataFrame()
    df["WAIS_COD_STD"] = WAIS_COD_STD[np.array(np.isnan(WAIS_COD_STD)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(WAIS_COD_STD)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(WAIS_COD_STD)==False)]
    mod = ols("U ~ WAIS_COD_STD +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"WAIS Picture Completion \n score effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["WAIS_COD_STD"],mod.pvalues["WAIS_COD_STD"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/WAIS_MC_TOT"
WAIS_MC_TOT = pop_vip["WAIS_MC_TOT"].values[y_vip==1]
for i in range(10):
    df = pd.DataFrame()
    df["WAIS_MC_TOT"] = WAIS_MC_TOT[np.array(np.isnan(WAIS_MC_TOT)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(WAIS_MC_TOT)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(WAIS_MC_TOT)==False)]
    mod = ols("U ~ WAIS_MC_TOT +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"WAIS Picture Completion \n score effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["WAIS_MC_TOT"],mod.pvalues["WAIS_MC_TOT"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/WAIS_ARITH_STD"
WAIS_ARITH_STD = pop_vip["WAIS_ARITH_STD"].values[y_vip==1]
for i in range(10):
    df = pd.DataFrame()
    df["WAIS_ARITH_STD"] = WAIS_ARITH_STD[np.array(np.isnan(WAIS_ARITH_STD)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(WAIS_ARITH_STD)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(WAIS_ARITH_STD)==False)]
    mod = ols("U ~ WAIS_ARITH_STD +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"WAIS Picture Completion \n score effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["WAIS_ARITH_STD"],mod.pvalues["WAIS_ARITH_STD"]))

    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/WAIS_ASS_OBJ_STD"
WAIS_ASS_OBJ_STD = pop_vip["WAIS_ASS_OBJ_STD"].values[y_vip==1]
for i in range(10):
    df = pd.DataFrame()
    df["WAIS_ASS_OBJ_STD"] = WAIS_ASS_OBJ_STD[np.array(np.isnan(WAIS_ASS_OBJ_STD)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(WAIS_ASS_OBJ_STD)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(WAIS_ASS_OBJ_STD)==False)]
    mod = ols("U ~ WAIS_ASS_OBJ_STD +age",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.6, .3,"WAIS Picture Completion \n score effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["WAIS_ASS_OBJ_STD"],mod.pvalues["WAIS_ASS_OBJ_STD"]))
    plt.figtext(0.6, .15,"AGE effect on U:\n Tvalue = %s \n pvalue = %s"
              %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)))

################################################################################
################################################################################
