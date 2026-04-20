"""
Univariate statistics for clinical variables.
"""


import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportions_ztest

import os
from config import data, config

# %% Utils for Univariate statistics
import itertools


def is_categorical(x, levels=2):
    """
    Guess if x is categorical.
    Returns True if x is not float and has at most 'levels' unique values.
    """
    return pd.Series(x).nunique(dropna=True) <= levels

def univ_stats(data, cols1, cols2, cattest="chi2"):

    res = list()

    for v1, v2 in itertools.product(cols1, cols2):
        df_ = data[[v1, v2]].dropna()
        # Check that all columns are numeric
        # df_.dtypes
        # df_.describe()
        if is_categorical(df_.values.ravel(), levels=2):
            if cattest == "prop_ztest": # proportions_ztest
                crosstab = pd.crosstab(df_[v1], df_[v2], rownames=[v1], colnames=[v2])
                zero_one, zero_sum = crosstab.iloc[0, 1], crosstab.iloc[0, :].sum()
                one_one, one_sum = crosstab.iloc[1, 1], crosstab.iloc[1, :].sum()
                pstat, pval = proportions_ztest(count=[zero_one, one_one], nobs=[zero_sum, one_sum], value=None, alternative='two-sided')
                stat, pval, dof, expected = scipy.stats.chi2_contingency(crosstab)
                dscr = str(crosstab.values).replace('\n', ' ') + " %s/%s" % (v1, v2)
                test = "prop_ztest"            
           
            else:  # Chi2
                crosstab = pd.crosstab(df_[v1], df_[v2], rownames=[v1], colnames=[v2])
                stat, pval, dof, expected = scipy.stats.chi2_contingency(crosstab)
                dscr = str(crosstab.values).replace('\n', ' ') + " %s/%s" % (v1, v2)
                test = "chi2"
            
        elif is_categorical(df_[v2].values, levels=2): # two-sample t-test / y
            l0, l1 = np.unique(df_[v2].values)
            x0, x1 = df_.loc[df_[v2] == l1, v1], df_.loc[df_[v2] == l0, v1]
            ttest = scipy.stats.ttest_ind(x0, x1, equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            dscr = "[%.3f  %.3f]" % (np.mean(x0), np.mean(x1))
            test = "ttest"
        
        elif is_categorical(df_[v1].values, levels=2): # two-sample t-test / x
            l0, l1 = np.unique(df_[v1].values)
            x0, x1 = df_.loc[df_[v1] == l1, v2], df_.loc[df_[v1] == l0, v2]
            ttest = scipy.stats.ttest_ind(x0, x1, equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            dscr = "[%.3f  %.3f]" % (np.mean(x0), np.mean(x1))
            test = "ttest"
        
        else:
            x0, x1 = df_[v1], df_[v2]
            test = scipy.stats.pearsonr(x0, x1)
            stat, pval = test.statistic, test.pvalue
            dscr = "[%.3f  %.3f]" % (np.mean(x0), np.mean(x1))
            test = "corr"
        
        res.append([v1, v2, dscr, stat, pval, test])

    res = pd.DataFrame(res, columns=['v1', 'v2', 'descriptive', 'stat', 'pval', 'test'])
    res = res.sort_values( 'pval')
    return(res)

# %% Descriptive statistics for clinical variables

data.to_csv(os.path.join(config['output_models'], 'data.csv'), index=False)
data.response.describe()
["Resp=%i, N=%i" % (resp, np.sum(data.response == resp)) for resp in data.response.unique()]
['Resp=1, N=24', 'Resp=0, N=34']

# %% Run Univariate statistics

# data.Catatonie[data.response == 0].mean()
# data.Catatonie[data.response == 1].mean()

stats = univ_stats(data, cols1=['response'], cols2=clinical_vars, cattest="prop_ztest")
stats.sort_values('pval', inplace=True)
stats.to_csv(os.path.join(config['output_models'], 'univariate_stats.csv'), index=False)

# %% Additional Statistics with propotion test 

v1, v2 ='DSM_TDAH', 'response'
df_ = data[[v1, v2]].dropna()
ct = pd.crosstab(df_[v1], df_[v2], rownames=[v1], colnames=[v2])
no_resp, no_sum = ct.iloc[0, 1], ct.iloc[0, :].sum()
yes_resp, yes_sum = ct.iloc[1, 1], ct.iloc[1, :].sum()
print("Prop No resp %.3f vs Prop Resp %.3f" % (no_resp / no_sum, yes_resp / yes_sum))


proportions_ztest(count=[no_resp, yes_resp], nobs=[no_sum, yes_sum], value=None, alternative='two-sided')
# (np.float64(-2.015767098021012), np.float64(0.0438243355622758))

proportions_ztest(count=[no_resp, yes_resp], nobs=[no_sum, yes_sum], value=None, alternative='smaller',)
# (np.float64(-2.015767098021012), np.float64(0.0219121677811379))

