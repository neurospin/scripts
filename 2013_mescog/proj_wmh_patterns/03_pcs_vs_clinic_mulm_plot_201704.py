# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:04:57 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause

rsync -avuh /neurospin/mescog/proj_wmh_patterns/struct_pca_0.003_0.003_0.003 /media/ed203246/usbed/neurospin/mescog/proj_wmh_patterns/
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

INPUT_MESCOG_DIR = "/neurospin/mescog/"

alpha, l1_ratio, l2_ratio, tv_ratio = 0.01, 1/3, 1/3, 1/3
ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
key = "struct_pca_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)

INPUT_DIR = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns", key)
INPUT_FILE = os.path.join(INPUT_DIR, "clinic_pc.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "stats.csv")


#################
# Merge PC with clinic
#################
clinic_pc = pd.read_csv(INPUT_FILE)
clinic = pd.read_csv(INPUT_CLINIC)

pc["ID"] = ["CAD_%i" % sid for sid in pc["Subject ID"]]

pc_cols = [col for col in pc.columns if col.count("pc")]

pc = pc[["ID"] + pc_cols]

data = pd.merge(clinic, pc, on="ID", how='outer')

COLNAMES_MAPPING = {'TMTB_TIME':'TMTB', "MDRS_TOTAL":"MDRS", "MRS": "mRS"}
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

PC_ALL = [x for x in data.columns if x.count("pc")]
PC_TVENET = [x for x in data.columns if # Of Interest
    (x.count('tvl1l2') and not x.count('tvl1l2_smalll1'))]
PC_TVENET.remove('pc_sum23__tvl1l2')

########
# Models
########

#formulas_allpcs_simple = ['%s~%s' % (t, r) for t in TARGETS_ALL for r in PC_ALL]

#formulas_clin_allpcs_democovars = ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
#        for t in TARGETS_CLIN for r in PC_ALL]

# Check associations between clinic and BPF and LLV
formulas_clin_nopcs_allcovars = \
    ['%s~BPF+LLV+WMHV+MBcount+AGE_AT_INCLUSION+SEX+EDUCATION' % t
        for t in TARGETS_CLIN]
contrasts = np.identity(7)[:4, ]
mod = MULM(data=data, formulas=formulas_clin_nopcs_allcovars)
stats_clin_nopcs_allcovars = mod.t_test(contrasts=contrasts.tolist(), out_filemane=None)


formulas_clin_allpcs_allcovars = \
    ['%s~%s+BPF+LLV+WMHV+MBcount+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_CLIN for r in PC_ALL]
contrasts = np.identity(8)[:5, ]
mod = MULM(data=data, formulas=formulas_clin_allpcs_allcovars)
stats_clin_allpcs_allcovars = mod.t_test(contrasts=contrasts.tolist(), out_filemane=None)

#formulas_allpcs = formulas_allpcs_simple + formulas_allpcs_democovars + formulas_allpcs_allcovars

formulas_clinbl_tvenet_allcovars = \
    ['%s~%s+BPF+LLV+WMHV+MBcount+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_CLIN_BL for r in PC_TVENET]
contrasts = np.identity(8)[:5, ]
mod = MULM(data=data, formulas=formulas_clinbl_tvenet_allcovars)
stats_clinbl_tvenet_allcovars = mod.t_test(contrasts=contrasts.tolist(), out_filemane=None)

formulas_ni_tvenet_democovars = \
    ['%s~%s+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_NI for r in PC_TVENET]
#contrasts = np.identity(4)[:3, ]
mod = MULM(data=data, formulas=formulas_ni_tvenet_democovars)
stats_ni_tvenet_democovars = mod.t_test(contrasts=1, out_filemane=None)

#mod = MULM(data=data, formulas=formulas_allpcs_allcovars)
#stats_allpcs_democovars = mod.t_test(contrasts=1, out_filemane=None)

#mod = MULM(data=data, formulas=formulas_all)
#stats_all = mod.t_test(contrasts=1, out_filemane=None)

summary_qc = stats_clin_nopcs_allcovars[stats_clin_nopcs_allcovars.target.isin(TARGETS_CLIN_BL) & 
                           stats_clin_nopcs_allcovars.contrast.isin(["WMHV", "MBcount"])]
summary_qc["pval_fdr"] = multipletests(summary_qc.pvalue,  method='fdr_bh')[1]
summary_qc.target.order()
summary_qc = summary_qc.sort(columns=['pval_fdr'])

# Summary association Clinical score with PC 
summary_pc = stats_clinbl_tvenet_allcovars.copy()
summary_pc = summary_pc[summary_pc.contrast.str.match("pc")]
summary_pc["pval_fdr"] = multipletests(summary_pc.pvalue,  method='fdr_bh')[1]
summary_pc.target = summary_pc.target.replace({'TMTB_TIME':'TMTB', "MDRS_TOTAL":"MDRS", "MRS": "mRS"})
summary_pc["PC"] = summary_pc.contrast.replace({'pc1__tvl1l2':1, 'pc2__tvl1l2':2, 'pc3__tvl1l2':3})
summary_pc = summary_pc.drop("contrast", axis=1)
summary_pc = summary_pc.ix[:, ["target", "PC", "tvalue", "pvalue", "pval_fdr"]]
    

with pd.ExcelWriter(os.path.join(OUTPUT, "pc_clinic_associations.xlsx")) as writer:
    summary_pc.to_excel(writer, sheet_name='summary PC', index=False)
    summary_qc.to_excel(writer, sheet_name='summary QC', index=False)
    stats_clin_nopcs_allcovars.to_excel(writer, sheet_name='clin_nopcs_allcovars', index=False)    
    stats_clin_allpcs_allcovars.to_excel(writer, sheet_name='clin_allpcs_allcovars', index=False)
    stats_clinbl_tvenet_allcovars.to_excel(writer, sheet_name='clinbl_tvenet_allcovars', index=False)
    stats_ni_tvenet_democovars.to_excel(writer, sheet_name='stats_ni_tvenet_democovars', index=False)
    data.to_excel(writer, sheet_name='data', index=False)
    
    #stats_all.to_excel(writer, sheet_name='all_simple+covars', index=False)



## OLDIES
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
    print(fig.get_figwidth())
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
                targets=TARGETS_ALL, regressors=PC_ALL,
                covar_models=COVARS, full_model=True)
stats_pcatv["method"] = "PCATV"

summary = stats_pcatv[
    (stats_pcatv.covariate.isin(PC_ALL))&
    stats_pcatv.target.isin(TARGETS_CLIN_BL+TARGETS_NI)]



stats_pca = mulm_df(data=methods["PCA"](data),
                targets=TARGETS_ALL, regressors=PC_ALL,
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