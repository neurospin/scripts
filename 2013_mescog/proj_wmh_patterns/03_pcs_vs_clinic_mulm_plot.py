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
from statsmodels.sandbox.stats.multicomp import multipletests
from patsy import dmatrices

BASE_DIR = "/home/ed203246/data/mescog/wmh_patterns"
INPUT_BASE_PC = INPUT_BASE_CLINIC = BASE_DIR

INPUT_PC = os.path.join(INPUT_BASE_PC, "summary", "components.csv")
# use the same file than in predcit cog decline
INPUT_CLINIC = os.path.join(INPUT_BASE_CLINIC, "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv")
OUTPUT = os.path.join(INPUT_BASE_PC, "summary")

COLNAMES_MAPPING = {'TMTB_TIME':'TMTB', "MDRS_TOTAL":"MDRS", "MRS": "mRS"}
data = pd.read_csv(INPUT_PC)
data.columns = [x.replace(".", "_") for x in data.columns]

#################
# Compute changes
#################

data["TMTB_TIME_CHANGE"] = data["TMTB_TIME_M36"] - data["TMTB_TIME"]
data["MDRS_TOTAL_CHANGE"] = data["MDRS_TOTAL_M36"] - data["MDRS_TOTAL"]
data["MRS_CHANGE"] = data["MRS_M36"] - data["MRS"]
data["MMSE_CHANGE"] = data["MMSE_M36"] - data["MMSE"]

###########
# Variables
###########

TARGETS_CLIN_BL = ["TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE"]
TARGETS_CLIN_M36 = ["TMTB_TIME_M36", "MDRS_TOTAL_M36", "MRS_M36", "MMSE_M36"]
TARGETS_CLIN_CHANGE = ["TMTB_TIME_CHANGE", "MDRS_TOTAL_CHANGE", "MRS_CHANGE", "MMSE_CHANGE"]
TARGETS_CLIN = TARGETS_CLIN_BL + TARGETS_CLIN_M36 + TARGETS_CLIN_CHANGE
TARGETS_NI = ['LLV', 'BPF', 'WMHV', 'MBcount']

TARGETS_ALL = TARGETS_CLIN + TARGETS_NI
TARGETS_OI = TARGETS_CLIN_BL + TARGETS_NI # Of Interest

REGRESSORS_ALL = [x for x in data.columns if x.count("pc")]
REGRESSORS_OI = [x for x in data.columns if # Of Interest
    (x.count('tvl1l2') and not x.count('tvl1l2_smalll1'))]
REGRESSORS_OI.remove('pc_sum23__tvl1l2')

########
# Models
########

formulas_all_simple = ['%s~%s' % (t, r) for t in TARGETS_ALL for r in REGRESSORS_ALL]
formulas_all_covars = \
    ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION+BPF+LLV' % (t, r)
        for t in TARGETS_CLIN for r in REGRESSORS_ALL] + \
    ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_NI for r in REGRESSORS_ALL]

formulas_all = formulas_all_simple + formulas_all_covars

formulas_oi = \
    ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION+BPF+LLV' % (t, r)
        for t in TARGETS_CLIN_BL for r in REGRESSORS_OI] + \
    ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_NI for r in REGRESSORS_OI]

mod = MULM(data=data, formulas=formulas_all_simple)
stats_all_simple = mod.t_test(contrasts=1, out_filemane=None)

mod = MULM(data=data, formulas=formulas_all_covars)
stats_all_covars = mod.t_test(contrasts=1, out_filemane=None)

mod = MULM(data=data, formulas=formulas_all)
stats_all = mod.t_test(contrasts=1, out_filemane=None)

mod = MULM(data=data, formulas=formulas_oi)
stats_oi = mod.t_test(contrasts=1, out_filemane=None)
stats_oi["Corrected P value"] = multipletests(stats_oi.pvalue,  method='fdr_bh')[1]


summary = stats_oi.copy()

summary["Variable"] = summary.target.replace({'TMTB_TIME':'TMTB', "MDRS_TOTAL":"MDRS", "MRS": "mRS"})
summary["PC"] = summary.contrast.replace({'pc1__tvl1l2':1, 'pc2__tvl1l2':2, 'pc3__tvl1l2':3})
summary["P value"] = summary.pvalue
summary["t statistic"] = summary.tvalue
summary = summary.loc[:, ["Variable", "PC", "t statistic", "P value", "Corrected P value"]]

with pd.ExcelWriter(os.path.join(OUTPUT, "pc_clinic_associations.xls")) as writer:
    summary.to_excel(writer, sheet_name='summary', index=False)
    stats_oi.to_excel(writer, sheet_name='models of interest', index=False)
    stats_all_simple.to_excel(writer, sheet_name='all_simple', index=False)
    stats_all_covars.to_excel(writer, sheet_name='all_covars', index=False)
    stats_all.to_excel(writer, sheet_name='all_simple+covars', index=False)


#################################
# Plots clinic by PC1, PC2, PC3
#################################

data.rename(columns=COLNAMES_MAPPING, inplace=True)
targets = [COLNAMES_MAPPING[k] for k in COLNAMES_MAPPING] + ["MMSE"]
data["PC1"]  = data.pc1__tvl1l2
data["PC2"]  = data.pc2__tvl1l2
data["PC3"]  = data.pc3__tvl1l2


PCS = [1, 2, 3]
pdf = PdfPages(os.path.join(OUTPUT, "pc_clinic_associations.pdf"))
for target in targets:
    #target = 'MMSE'
    #target = 'TMTB'
    dt = data[data[target].notnull()]
    y = dt[target]
    fig, axarr = plt.subplots(1, 3)#, sharey=True)
    fig.set_figwidth(15)
    print fig.get_figwidth()
    for j, pc in enumerate(PCS):
        #j, pc = 2, 3
        # --------------------------------
        model = '%s~PC%s+AGE_AT_INCLUSION+SEX+EDUCATION+BPF+LLV' % (target, pc)
        # --------------------------------
        y, X = dmatrices(model, data=dt, return_type='dataframe')
        mod = sm.OLS(y, X).fit()
        test = mod.t_test([0, 1]+[0]*(X.shape[1]-2))
        tval, pval =  test.tvalue[0, 0], test.pvalue[0, 0]
        x = dt["PC%i" % pc]
        axarr[j].scatter(x, y)
        if False:
            for i in xrange(len(dt['Subject ID'])):
                axarr[j].text(dt.ix[i, "PC%i" % pc], y.ix[i,0], dt['Subject ID'][i])
        x_ext = np.array([x.min(), x.max()])
        y_ext = x_ext * mod.params[1] + y.mean().values#mod.params[0]
        axarr[j].plot(x_ext, y_ext, "red")
        if j == 0:
            axarr[j].set_ylabel(target)
        axarr[j].set_xlabel('PC%i (T=%.3f, P=%.4g)' % (pc, tval, pval))
        #axarr[j].set_xticklabels([])
    fig.suptitle(target, size='large')
    fig.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

pdf.close()

#################################
# Plots PC1 x PC3 color by clinic
#################################

pdf = PdfPages(os.path.join(OUTPUT, "pc2-pc3_clinic_color.pdf"))
#pdf = PdfPages(os.path.join(OUTPUT, "pc1-pc3_clinic_color.pdf"))
for target in targets:
    #target = 'MMSE'
    #target = 'TMTB'
    dt = data[data[target].notnull()]
    y = dt[target]
    fig = plt.figure()#, sharey=True)
    #fig.set_figwidth(15)
    print fig.get_figwidth()
    #dt.PC1, dt.PC3
    plt.scatter(dt.PC2, dt.PC3, c=dt[target], s=50)
    if False:
        for i in xrange(len(dt['Subject ID'])):
            plt.text(dt.ix[i, "PC%i" % pc], y.ix[i,0], dt['Subject ID'][i])
    plt.colorbar()
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    #axarr[j].set_xticklabels([])
    fig.suptitle(target, size='large')
    fig.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

pdf.close()












x = dt["PC%i" % 3]
plt.scatter(x, y)
ids = [str(s) for s in dt['Subject ID']]

plt.scatter(x, y)
plt.text(x, y, ids)
plt.show()

stats_oi.target

x = data.pc1__tvl1l2
y = data.pc3__tvl1l2

plt.scatter(x, y)
plt.scatter(x, y, c=data.TMTB_TIME , s=50)

plt.scatter(x=y, y=data.TMTB_TIME , s=50)

plt.scatter(PCs[:, 0], PCs[:, 1], s=50)#, "ob", s=50)
for i in xrange(len(moments["lacune_id"])):
    plt.text(PCs[i, 0], PCs[i, 1], moments["lacune_id"][i])

"""
stats_pcatv = mulm_df(data=methods["PCATV"](data),
                targets=TARGETS_ALL, regressors=REGRESSORS_ALL,
                covar_models=COVARS, full_model=True)
stats_pcatv["method"] = "PCATV"

summary = stats_pcatv[
    (stats_pcatv.covariate.isin(REGRESSORS_ALL))&
    stats_pcatv.target.isin(TARGETS_CLIN_BL+TARGETS_NI)]



stats_pca = mulm_df(data=methods["PCA"](data),
                targets=TARGETS_ALL, regressors=REGRESSORS_ALL,
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
for target in TARGETS_ALL:
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