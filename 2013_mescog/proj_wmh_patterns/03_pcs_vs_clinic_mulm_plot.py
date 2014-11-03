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
from patsy import dmatrices
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#INPUT_BASE_CLINIC = "/neurospin/mescog/proj_predict_cog_decline/data"
#INPUT_BASE_PC = "/neurospin/brainomics/2014_pca_struct/mescog/mescog_5folds"
BASE_DIR = "/home/ed203246/data/mescog/wmh_patterns"
INPUT_BASE_PC = INPUT_BASE_CLINIC = BASE_DIR

INPUT_PC = os.path.join(INPUT_BASE_PC, "data", "components.csv")
# use the same file than in predcit cog decline
INPUT_CLINIC = os.path.join(INPUT_BASE_CLINIC, "data", "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv")
OUTPUT = os.path.join(INPUT_BASE_PC, "results") 

pc = pd.read_csv(INPUT_PC)
clinic = pd.read_csv(INPUT_CLINIC)

clinic["ID"] = [int(x.replace("CAD_", "")) for x in clinic.ID.tolist()]
clinic.SEX -= 1

data = pd.merge(clinic, pc, left_on="ID", right_on="Subject ID")
data.columns = [x.replace(".", "_") for x in data.columns]

def get_pca(d):
    return d[(d.global_pen == 0) & (d.tv_ratio == 0) & (d.l1_ratio == 0)]

def get_l1l2tv(d):
    return d[(d.global_pen == 1) & (d.tv_ratio == .33) & (d.l1_ratio == .5)]

methods = dict(PCATV=get_l1l2tv, PCA=get_pca)

assert get_pca(data).shape == get_l1l2tv(data).shape == (301, 33)


data["TMTB_TIME_CHANGE"] = data["TMTB_TIME_M36"] - data["TMTB_TIME"]
data["MDRS_TOTAL_CHANGE"] = data["MDRS_TOTAL_M36"] - data["MDRS_TOTAL"]
data["MRS_CHANGE"] = data["MRS_M36"] - data["MRS"]
data["MMSE_CHANGE"] = data["MMSE_M36"] - data["MMSE"]

TARGETS = [
           "TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE",
           "TMTB_TIME_M36", "MDRS_TOTAL_M36", "MRS_M36", "MMSE_M36",
           "TMTB_TIME_CHANGE", "MDRS_TOTAL_CHANGE", "MRS_CHANGE", "MMSE_CHANGE"]

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
            res.append([method, target, pc, model, tval, pval])
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
