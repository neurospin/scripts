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

data = pd.merge(clinic, pc, left_on="ID", right_on="Subject ID")

def get_pca(d):
    return d[(d.global_pen == 0) & (d.tv_ratio == 0) & (d.l1_ratio == 0)]

def get_l1l2tv(d):
    return d[(d.global_pen == 1) & (d.tv_ratio == .33) & (d.l1_ratio == .5)]

models = dict(PCATV=get_l1l2tv, PCA=get_pca)

assert get_pca(data).shape == get_l1l2tv(data).shape == (301, 33)


data["TMTB_TIME.CHANGE"] = data["TMTB_TIME.M36"] - data["TMTB_TIME"]
data["MDRS_TOTAL.CHANGE"] = data["MDRS_TOTAL.M36"] - data["MDRS_TOTAL"]
data["MRS.CHANGE"] = data["MRS.M36"] - data["MRS"]
data["MMSE.CHANGE"] = data["MMSE.M36"] - data["MMSE"]

TARGETS = ["TMTB_TIME.M36", "MDRS_TOTAL.M36", "MRS.M36", "MMSE.M36",
           "TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE",
           "TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE"]
import statsmodels.graphics as smg

pdf = PdfPages(os.path.join(OUTPUT, "pc_clinic_associations.pdf"))

PCS = [2, 3]
res = list()
for target in TARGETS:
    dt = data[data[target].notnull()]
    fig, axarr = plt.subplots(2, 2)#, sharey=True)
    for i, model in enumerate(models):
        d = models[model](dt)
        X = np.ones((d.shape[0], 2))
        for j, pc in enumerate(PCS):
            y = d[target]
            x = d["PC%i" % pc]
            X[:, 0] = x
            mod = sm.OLS(y, X)
            sm_fitted = mod.fit()
            sm_ttest = sm_fitted.t_test([1, 0])
            tval, pval =  sm_ttest.tvalue[0, 0], sm_ttest.pvalue[0, 0]
            res.append([model, target, pc, tval, pval])
            #axarr[i, j].set_title('%s (T=%.3f, P=%.4g)' % (model, tval, pval))
            if i == 0:
                axarr[i, j].set_title("PC%i" % pc)
            axarr[i, j].scatter(x, y)
            axarr[i, j].plot(x, sm_fitted.fittedvalues)
            axarr[i, j].set_xlabel('T=%.3f, P=%.4g' % (tval, pval))
            if j == 0:
                axarr[i, j].set_ylabel(model, rotation=0, size='large')
            axarr[i, j].set_xticklabels([])
    fig.suptitle(target, size='large')
    fig.tight_layout()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()
    
pdf.close()


stats = pd.DataFrame(res, columns=["model", "var", "pc", "tval", "pval"])

print stats
stats.to_csv(os.path.join(OUTPUT, "pc_clinic_associations.csv"), index=False)

"""
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.scatter(1,1)
ax2 = fig.add_subplot(2,1,2,sharex=ax1)
ax2.scatter(1,1)


#[left, bottom, width, height]
ax = fig.add_axes( [0., 0., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text( 
    .05, 0.5, "left Label", rotation='vertical',
    horizontalalignment='center', verticalalignment='center'
)

#[left, bottom, width, height]
ax = fig.add_axes( [0., -1., 1, 1] )
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.text( 
    .05, 0.5, "top Label",
    horizontalalignment='center', verticalalignment='center'
)
plt.show()
"""