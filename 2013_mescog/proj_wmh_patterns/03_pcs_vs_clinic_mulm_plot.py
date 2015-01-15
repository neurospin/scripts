# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:04:57 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
#from patsy import dmatrices
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mulm.dataframe.descriptive_statistics import describe_df_basic
from mulm.dataframe.mulm_dataframe import MULM

#INPUT_BASE_CLINIC = "/neurospin/mescog/proj_predict_cog_decline/data"
#INPUT_BASE_PC = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
BASE_DIR = "/home/ed203246/data/mescog/wmh_patterns"
INPUT_BASE_PC = INPUT_BASE_CLINIC = BASE_DIR

INPUT_PC = os.path.join(INPUT_BASE_PC, "summary", "components.csv")
# use the same file than in predcit cog decline
INPUT_CLINIC = os.path.join(INPUT_BASE_CLINIC, "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv")
OUTPUT = os.path.join(INPUT_BASE_PC, "summary")

data = pd.read_csv(INPUT_PC)

"""
clinic = pd.read_csv(INPUT_CLINIC)

clinic["ID"] = [int(x.replace("CAD_", "")) for x in clinic.ID.tolist()]
clinic.SEX -= 1

data = pd.merge(clinic, pc, left_on="ID", right_on="Subject ID")


def get_pca(d):
    return d[(d.global_pen == 0) & (d.tv_ratio == 0) & (d.l1_ratio == 0)]

def get_l1l2tv(d):
    d = d[(d.global_pen == 1) & (d.tv_ratio == .33) & (d.l1_ratio == .5)]
    d.PC1 = -d.PC1  # get the opposite value of PC1 to harmonize
    return d

methods = dict(PCATV=get_l1l2tv, PCA=get_pca)

assert get_pca(data).shape == get_l1l2tv(data).shape == (301, 33)
"""

#data = pc
data.columns = [x.replace(".", "_") for x in data.columns]

# Changes
data["TMTB_TIME_CHANGE"] = data["TMTB_TIME_M36"] - data["TMTB_TIME"]
data["MDRS_TOTAL_CHANGE"] = data["MDRS_TOTAL_M36"] - data["MDRS_TOTAL"]
data["MRS_CHANGE"] = data["MRS_M36"] - data["MRS"]
data["MMSE_CHANGE"] = data["MMSE_M36"] - data["MMSE"]

TARGETS_CLIN_BL = ["TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE"]
TARGETS_CLIN_M36 = ["TMTB_TIME_M36", "MDRS_TOTAL_M36", "MRS_M36", "MMSE_M36"]
TARGETS_CLIN_CHANGE = ["TMTB_TIME_CHANGE", "MDRS_TOTAL_CHANGE", "MRS_CHANGE", "MMSE_CHANGE"]
TARGETS_NI_GLOB = ['LLV', 'BPF', 'WMHV', 'MBcount']
TARGETS = TARGETS_CLIN_BL + TARGETS_CLIN_M36 + TARGETS_CLIN_CHANGE + TARGETS_NI_GLOB
TARGETS_SUMMARY = TARGETS_CLIN_BL + TARGETS_NI_GLOB
REGRESSORS = [x for x in data.columns if x.count("pc")]
REGRESSORS_SUMMARY = [x for x in data.columns if
    (x.count("pca") or (x.count('tvl1l2') and not x.count('tvl1l2_smalll1')))]

formulas_simple = ['%s~%s' % (t, r) for t in TARGETS for r in REGRESSORS]
formulas_covars = [f +  "+AGE_AT_INCLUSION+SEX+EDUCATION" for f in formulas_simple]
formulas_all = formulas_simple + formulas_covars
formulas_covars_summary = \
['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
    for t in TARGETS_SUMMARY for r in REGRESSORS_SUMMARY]


model_simple = MULM(data=data, formulas=formulas_simple)
stats_simple = model_simple.t_test(contrasts=1, out_filemane=None)

model_covars = MULM(data=data, formulas=formulas_covars)
stats_covars = model_covars.t_test(contrasts=1, out_filemane=None)

model_all = MULM(data=data, formulas=formulas_all)
stats_all = model_all.t_test(contrasts=1, out_filemane=None)

model_summary = MULM(data=data, formulas=formulas_covars_summary)
stats_summary = model_summary.t_test(contrasts=1, out_filemane=None)

# -
formulas_covars_summary_sumpc12 = \
['%s~pc2__tvl1l2+pc3__tvl1l2+AGE_AT_INCLUSION+SEX+EDUCATION' % t for t in TARGETS_SUMMARY]

model_summary_sumpc12 = MULM(data=data, formulas=formulas_covars_summary_sumpc12)
stats_summary_sumpc12 = model_summary_sumpc12.t_test(contrasts=1, out_filemane=None)

with pd.ExcelWriter(os.path.join(OUTPUT, "pc_clinic_associations.xls")) as writer:
    stats_simple.to_excel(writer, sheet_name='simple', index=False)
    stats_covars.to_excel(writer, sheet_name='covars', index=False)
    stats_all.to_excel(writer, sheet_name='all', index=False)
    stats_summary.to_excel(writer, sheet_name='summary', index=False)









"""
stats_pcatv = mulm_df(data=methods["PCATV"](data),
                targets=TARGETS, regressors=REGRESSORS,
                covar_models=COVARS, full_model=True)
stats_pcatv["method"] = "PCATV"

summary = stats_pcatv[
    (stats_pcatv.covariate.isin(REGRESSORS))&
    stats_pcatv.target.isin(TARGETS_CLIN_BL+TARGETS_NI_GLOB)]



stats_pca = mulm_df(data=methods["PCA"](data),
                targets=TARGETS, regressors=REGRESSORS,
                covar_models=COVARS, full_model=True)
stats_pca["method"] = "PCA"



with pd.ExcelWriter(os.path.join(OUTPUT, "pc_clinic_associations.xls")) as writer:
    stats_pcatv.to_excel(writer, sheet_name='PCATV', index=False)
    summary.to_excel(writer, sheet_name='PCATV short', index=False)
    stats_pca.to_excel(writer, sheet_name='PCA', index=False)
"""

"""
pdf = PdfPages(os.path.join(OUTPUT, "pc_clinic_associations.pdf"))
#print os.path.join(OUTPUT, "pc_clinic_associations.pdf")

PCS = [1, 2, 3]
res = list()
for target in TARGETS:
    dt = data[data[target].notnull()]
    fig, axarr = plt.subplots(2, 3)#, sharey=True)
    for i, method in enumerate(methods):
        d = methods[method](dt)
        #X = np.ones((d.shape[0], 2))
        # --------------------------------
        model = '%s~PC1+PC2+PC3' % target
        # --------------------------------
        y, X = dmatrices(model, data=d, return_type='dataframe')
        mod = sm.OLS(y, X)
        sm_fitted = mod.fit()
        sm_ttest = sm_fitted.t_test([0, 1, 1, 1])
        tval, pval =  sm_ttest.tvalue[0, 0], sm_ttest.pvalue[0, 0]
        res.append([method, target, "PC1+PC2+PC3", model, tval, pval])
        for j, pc in enumerate(PCS):
            # --------------------------------
            model = '%s~PC%i' % (target, pc)
            # --------------------------------
            y, X = dmatrices(model, data=d, return_type='dataframe')
            mod = sm.OLS(y, X)
            sm_fitted = mod.fit()
            sm_ttest = sm_fitted.t_test([0, 1])
            tval, pval =  sm_ttest.tvalue[0, 0], sm_ttest.pvalue[0, 0]
            res.append([method, target, "PC%i" % pc, model, tval, pval])
            if i == 0:
                axarr[i, j].set_title("PC%i" % pc)
            axarr[i, j].scatter(d["PC%i" % pc], y)
            axarr[i, j].plot(d["PC%i" % pc], sm_fitted.fittedvalues, "black")
            axarr[i, j].set_xlabel('T=%.3f, P=%.4g' % (tval, pval))
            if j == 0:
                axarr[i, j].set_ylabel(method, rotation=0, size='large')
            axarr[i, j].set_xticklabels([])
            # --------------------------------
            model = '%s~PC%i+AGE_AT_INCLUSION+SEX+EDUCATION+SITE' % (target, pc)
            # --------------------------------
            y, X = dmatrices(model, data=d, return_type='dataframe')
            mod = sm.OLS(y, X)
            sm_fitted = mod.fit()
            sm_ttest = sm_fitted.t_test([0, 1] + [0] * (X.shape[1] - 2))
            tval, pval =  sm_ttest.tvalue[0, 0], sm_ttest.pvalue[0, 0]
            res.append([method, target, pc, model, tval, pval])
    fig.suptitle(target, size='large')
    fig.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

pdf.close()


stats = pd.DataFrame(res, columns=["method", "var", "pc", "model", "tval", "pval"])

stats.to_csv(os.path.join(OUTPUT, "pc_clinic_associations.csv"), index=False)
"""