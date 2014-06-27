# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:53:58 2014

@author: edouard.duchesnay@cea.fr
"""
import os
import pandas as pd
import numpy as np
import copy

params_columns = ["a", "l1", "l2", "tv"]
input_xls_filename = os.path.join(os.environ["HOME"],
        "Dropbox/results/adni/all_tvenet_results.xlsx")
output_summary_filename = os.path.join(os.environ["HOME"],
        "Dropbox/results/adni/all_tvenet_results_summary.csv")

def get_params(l1_ratio, tv_ratio):
    l1 = float(1 - tv_ratio) * l1_ratio
    l2 = float(1 - tv_ratio) * (1 - l1_ratio)
    return l1, l2, tv_ratio

def summarize(data, alpha, l1_ratios, tv_ratios):
    params = pd.DataFrame([[float(p) for p in item.split("_")] for item in data.key],
                            columns = params_columns)
    data = pd.concat([data, params], axis=1)
    #tmp = copy.copy(params_columns)
    #tmp.remove("a")
    data = data[data.a == alpha]
    #data = data[data.a == alpha][tmp + ['recall_mean']]
    # Choose on l1 and compare
    #d = data
    #l1_ratio = 0.1
    #tv = 0.01
    ret =None
    for l1_ratio in l1_ratios:
        for tv_i in xrange(len(tv_ratios)):
            l1, l2, tv = get_params(l1_ratio, tv_ratios[tv_i])
            mask = (np.abs(data.l1 - l1) <1e-6) & (np.abs(data.l2 - l2) <1e-6) & (np.abs(data.tv - tv) <1e-6)
            print l1, l2, tv, mask.sum()
            if ret is None and mask.sum():
                ret = data[mask]
            elif mask.sum():
                ret = ret.append(data[mask])
    return ret

alpha = 0.01
l1_ratios = np.array([.0, .001, .01, .1])
tv_ratios = np.array([.0, .001, .01, .1])
dataset = 'MCIc-MCInc'
data = pd.read_excel(input_xls_filename, dataset)#, index_col=None, na_values=['NA'])
s1 = summarize(data, alpha, l1_ratios, tv_ratios)
s1["dataset"] = [dataset] * len(s1)

dataset = 'MCIc-CTL'
data = pd.read_excel(input_xls_filename, dataset)#, index_col=None, na_values=['NA'])
s2 = summarize(data, alpha, l1_ratios, tv_ratios)
s2["dataset"] = [dataset] * len(s2)

dataset = 'AD-CTL'
data = pd.read_excel(input_xls_filename, dataset)#, index_col=None, na_values=['NA'])
s3 = summarize(data, alpha, l1_ratios, tv_ratios)
s3["dataset"] = [dataset] * len(s3)

summary = s1.append(s2)
summary = summary.append(s3)

summary.to_csv(output_summary_filename, index=False)
