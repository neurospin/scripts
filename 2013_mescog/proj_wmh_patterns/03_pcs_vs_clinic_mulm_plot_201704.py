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

if False:  # Parameters settings 3
    alpha, l1_ratio, l2_ratio, tv_ratio = 0.01, 1/3, 1/3, 1/3
    ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
    key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)

if False:  # Parameters settings 3
    #  1/3, 1/3 1/3 such that ll1 < l1max
    alpha, l1_ratio, l2_ratio, tv_ratio = 1., 0.1 * 0.025937425654559931, 1/3, 1/3
    ll1, ll2, ltv = alpha * l1_ratio, alpha * l2_ratio, alpha * tv_ratio
    key_pca_enettv_01l1max = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)
    key_pca_enettv = key_pca_enettv_01l1max

ll1, ll2, ltv = 0.1 * 0.025937425654559931, 1/3, 0.01 * 1/3
key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f" % (ll1, ll2, ltv)

ll1, ll2, ltv =  0.1 * 0.025937425654559931, 1/3, 0.01 * 1/3
key_pca_enettv = "pca_enettv_%.3f_%.3f_%.3f_inner_max_iter100" % (ll1, ll2, ltv)

ll1, ll2, ltv = 0.01 * 0.025937425654559931, 1/3, 0.01 * 1/3
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)

ll1, ll2, ltv = 0.1 * 0.025937425654559931, 1, 0.1
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)

ll1, ll2, ltv = 0.05 * 0.025937425654559931, 1, 0.001
key_pca_enettv = "pca_enettv_%.4f_%.3f_%.3f" % (ll1, ll2, ltv)
CHOICE = key_pca_enettv


key_pca = "pca"
key = CHOICE

#key = key_pca

INPUT_DIR = os.path.join(INPUT_MESCOG_DIR, "proj_wmh_patterns", "PCs", key)
INPUT_FILE = os.path.join(INPUT_DIR, "clinic_pc.csv")
OUTPUT_FILE = os.path.join(INPUT_DIR, "stats.xlsx")


#################
# Merge PC with clinic
#################
data = pd.read_csv(INPUT_FILE)
"""
clinic = pd.read_csv(INPUT_CLINIC)

pc["ID"] = ["CAD_%i" % sid for sid in pc["Subject ID"]]

pc_cols = [col for col in pc.columns if col.count("pc")]

pc = pc[["ID"] + pc_cols]

data = pd.merge(clinic, pc, on="ID", how='outer')
"""

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

"""
PC_ALL = [x for x in data.columns if x.count("PC")]
PC_TVENET = [x for x in data.columns if # Of Interest
    (x.count('tvl1l2') and not x.count('tvl1l2_smalll1'))]
PC_TVENET.remove('pc_sum23__tvl1l2')
"""
PC_TVENET = [x for x in data.columns if x.count("PC")]
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

"""
formulas_clin_allpcs_allcovars = \
    ['%s~%s+BPF+LLV+WMHV+MBcount+AGE_AT_INCLUSION+SEX+EDUCATION' % (t, r)
        for t in TARGETS_CLIN for r in PC_ALL]
contrasts = np.identity(8)[:5, ]
mod = MULM(data=data, formulas=formulas_clin_allpcs_allcovars)
stats_clin_allpcs_allcovars = mod.t_test(contrasts=contrasts.tolist(), out_filemane=None)
"""

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
summary_pc = summary_pc[summary_pc.contrast.str.match("PC")]
summary_pc["pval_fdr"] = multipletests(summary_pc.pvalue,  method='fdr_bh')[1]
summary_pc.target = summary_pc.target.replace({'TMTB_TIME':'TMTB', "MDRS_TOTAL":"MDRS", "MRS": "mRS"})
summary_pc["PC"] = summary_pc.contrast.replace({'PC0':0, 'PC1':1, 'PC2':2, 'PC3':3, 'PC4':4, 'PC5':5, 'PC6':6, 'PC7':7, 'PC8':8, 'PC9':9, 'PC10':10})
summary_pc = summary_pc.drop("contrast", axis=1)
summary_pc = summary_pc.ix[:, ["target", "PC", "tvalue", "pvalue", "pval_fdr", "formula"]]


with pd.ExcelWriter(OUTPUT_FILE) as writer:
    summary_pc.to_excel(writer, sheet_name='PC vs clinic', index=False)
    #summary_qc.to_excel(writer, sheet_name='summary QC', index=False)
    #stats_clin_nopcs_allcovars.to_excel(writer, sheet_name='clin_nopcs_allcovars', index=False)
    #stats_clinbl_tvenet_allcovars.to_excel(writer, sheet_name='clinbl_tvenet_allcovars', index=False)
    #stats_ni_tvenet_democovars.to_excel(writer, sheet_name='stats_ni_tvenet_democovars', index=False)
    data.to_excel(writer, sheet_name='data', index=False)

    #stats_all.to_excel(writer, sheet_name='all_simple+covars', index=False)

