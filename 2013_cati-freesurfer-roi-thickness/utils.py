# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:19:33 2013

@author: ed203246
"""

import numpy as np
import pandas as pd


def dataframe_quality_control(df):
    """Quality control of a data frame. Split Numerical, categorial columns and
    check for missing value

    Parameter
    ---------
        df: pandas DataFrame

    Return
    ------
    Four pandas DataFrame, numerical, non_numerical, withNAs, stats_d
    """
    non_num_cols = list()
    num_cols = list()
    qc_stats = list()

    for col in df.columns:
        v = df[col]
        if not np.issubdtype(v.dtype, float) and \
           not np.issubdtype(v.dtype, int):
            non_num_cols.append(col)
            qc_stats.append((col, "cat", None, None, None, None, None,
                             np.sum(v.isnull())))
        else:
            num_cols.append(col)
            mean = np.mean(v)
            std = np.std(v)
            nb_outliers = np.sum(np.abs(v - mean) > 4 * std)
            qc_stats.append((col, "num", np.min(v), np.max(v),
                             mean, std, nb_outliers, sum(np.isnan(v))))

    qc_stats = pd.DataFrame.from_records(qc_stats,
                                          columns=["col", "type", "min", "max",
                                                   "mean", "std",
                                                    "nb_outliers", "nb_NAs"])

    # Remove column with NAs
    na_cols = df.columns[qc_stats.nb_NAs != 0]
    for col in na_cols:
        if col in non_num_cols:
            non_num_cols.remove(col)
        if col in num_cols:
            num_cols.remove(col)

    # split dataset into numerical, categorial, withNAs
    d_num = df[num_cols]
    d_cat = df[non_num_cols]
    d_nas = df[na_cols]

    # Final check, try to cast in numerical array
    X = np.asarray(d_num)
    if not np.issubdtype(X.dtype, float) and not np.issubdtype(X.dtype, int):
        raise ValueError("Could not cast numerical column in numerical array")

    return d_num, d_cat, d_nas, qc_stats