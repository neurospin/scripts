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
sex = pop_vip["sex_code"].values


scores = "CVLT_RC_listeA_tot","CVLT_P_listeA_tot","CVLT_I_listeA_tot",\
"CVLT_RICT_RC_tot","CVLT_RICT_P_tot","CVLT_RICT_I_tot","CVLT_RICT_I_tot",\
"WAIS_COMPL_IM_STD","WAIS_COMPL_IM_CR","WAIS_VOC_TOT","WAIS_VOC_STD",\
"WAIS_VOC_CR","WAIS_COD_tot","WAIS_COD_err","WAIS_COD_brut","WAIS_COD_CR",\
"WAIS_COD_STD","WAIS_SIMI_tot","WAIS_SIMI_STD","WAIS_SIMI_CR","NART33_Tot",\
"NART33_QIT","NART33_QIV","NART33_QIP","WAIS_CUB_TOT","WAIS_CUB_STD",\
"WAIS_CUB_CR","WAIS_ARITH_T0T","WAIS_ARITH_STD","WAIS_ARITH_CR","WAIS_MC_OD_TOT",\

scores = "WAIS_MC_OINV_TOT","WAIS_MC_TOT","WAIS_MC_EMP_END","WAIS_MC_EMP_ENV","WAIS_MC_STD",\
"WAIS_MC_CR","WAIS_MC_EMP_END_STD","WAIS_MC_EMP_ENV_STD","WAIS_INFO_TOT","WAIS_INFO_STD",\
"WAIS_INFO_CR","WAIS_SLC_TOT","WAIS_SLC_STD","WAIS_SLC_CR","WAIS_ASS_OBJ_TOT","WAIS_ASS_OBJ_STD",\
"WAIS_ASS_OBJ_CR","WAIS_DET_MENT",

# Turn interactive plotting off
plt.ioff()

WD = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/scores"
for s in scores:
    output = os.path.join(WD,s)
    if os.path.exists(output) == False:
        os.makedirs(output)
        neurospycho = pop_vip[s].values[y_vip==1]
        for i in range(10):
            df = pd.DataFrame()
            df["neurospycho"] = neurospycho [np.array(np.isnan(neurospycho)==False)]
            df["age"] = age[y_vip==1][np.array(np.isnan(neurospycho)==False)]
            df["sex"] = sex[y_vip==1][np.array(np.isnan(neurospycho)==False)]
            df["U"] = U_vip_scz[:,i][np.array(np.isnan(neurospycho)==False)]
            mod = ols("U ~ neurospycho +age + sex",data = df).fit()
            print(mod.summary())
            fig = plt.figure(figsize=(10,6))
            fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
            plt.figtext(0.1,-0.1,"neurospycho effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["neurospycho"],mod.pvalues["neurospycho"]))

            plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
            plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
            plt.tight_layout()
            plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
            plt.close(fig)



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
    df["sex"] = sex[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_pos +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"panss_pos effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["panss_pos"],mod.pvalues["panss_pos"]))
    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.tight_layout()


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/panss_neg"
for i in range(10):
    df = pd.DataFrame()
    df["panss_neg"] = panss_neg_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["sex"] = sex[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_neg +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"panss_neg effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["panss_neg"],mod.pvalues["panss_neg"]))
    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.tight_layout()



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/panss_comp"
for i in range(10):
    df = pd.DataFrame()
    df["panss_comp"] = panss_comp_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["sex"] = sex[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ panss_comp +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"panss_comp effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["panss_comp"],mod.pvalues["panss_comp"]))
    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.tight_layout()


output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/fast"
for i in range(10):
    df = pd.DataFrame()
    df["fast"] = fast_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["sex"] = sex[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ fast +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"fast effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["fast"],mod.pvalues["fast"]))
    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.tight_layout()



output = "/neurospin/brainomics/2016_schizConnect/analysis/all_studies+VIP/VBM/\
all_subjects/results/pcatv_scz/results/projection_vip/hallu_panssP3"
for i in range(10):
    df = pd.DataFrame()
    df["hallu_panssP3"] = hallu_panssP3_scz[np.array(np.isnan(panss_pos_scz)==False)]
    df["age"] = age[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["sex"] = sex[y_vip==1][np.array(np.isnan(panss_pos_scz)==False)]
    df["U"] = U_vip_scz[:,i][np.array(np.isnan(panss_pos_scz)==False)]
    mod = ols("U ~ hallu_panssP3 +age+sex",data = df).fit()
    #print(mod.summary())
    fig = plt.figure(figsize=(10,6))
    fig = sm.graphics.plot_partregress_grid(mod, fig=fig)
    plt.figtext(0.1,-0.1,"hallu_panssP3 effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["hallu_panssP3"],mod.pvalues["hallu_panssP3"]))
    plt.figtext(0.7, -0.1,"age effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["age"],mod.pvalues["age"]))
    plt.figtext(0.4,-0.1,"sex effect on U:\n Tvalue = %s \n pvalue = %s"
                  %(mod.tvalues["sex"],mod.pvalues["sex"]))
    plt.tight_layout()
    plt.savefig(os.path.join(output,"comp%s"%(i+1)),bbox_inches = 'tight')
    plt.tight_layout()



################################################################################
################################################################################

