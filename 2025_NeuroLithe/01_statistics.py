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

def univ_stats_backup(data, cols1, cols2, cattest="chi2"):

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

def _var_type(s: pd.Series, max_cat_unique: int = 10) -> str:
    """
    Classify a series as 'constant', 'binary', 'multicategory', or 'quantitative'.

    Float columns are always quantitative.
    Integer / object columns with <= max_cat_unique distinct values are categorical.
    """
    n = s.nunique(dropna=True)
    if n <= 1:
        return 'constant'
    if n == 2:
        return 'binary'
    if s.dtype.kind == 'f' or n > max_cat_unique:
        return 'quantitative'
    return 'multicategory'


def _group_stats(quant: pd.Series, cat: pd.Series) -> str:
    """'level: mean±sd (n)' for each level of cat, joined by ' | '."""
    parts = []
    for lev in sorted(cat.unique()):
        g = quant[cat == lev]
        parts.append(f"{lev}: {g.mean():.3f}±{g.std():.3f} (n={len(g)})")
    return " | ".join(parts)


def univ_stats(data: pd.DataFrame, vars1: list, vars2: list,
               cattest: str = "chi2", max_cat_unique: int = 10) -> pd.DataFrame:
    """
    Pairwise statistical association for each (v1, v2) in vars1 × vars2.

    Test selection (grouping vs outcome resolved automatically):
      quant × quant          → Pearson r   (stat=r,    dof=n-2)
      quant × binary         → Welch t     (stat=t,    dof=Welch df,    means±sd per group)
      quant × multicategory  → one-way F   (stat=F,    dof=(k-1, n-k), means±sd per group)
      categ × categ          → chi2        (stat=chi2, dof=(r-1)(c-1), row proportions)
                               or prop_ztest when cattest='prop_ztest' and both are binary

    Parameters
    ----------
    max_cat_unique : int
        Integer columns with <= this many distinct values are treated as categorical.
    """
    rows = []

    for v1, v2 in itertools.product(vars1, vars2):
        if v1 == v2:
            continue
        df_ = data[[v1, v2]].dropna()
        if len(df_) < 4:
            continue

        s1, s2 = df_[v1], df_[v2]
        t1 = _var_type(s1, max_cat_unique)
        t2 = _var_type(s2, max_cat_unique)

        if 'constant' in (t1, t2):
            continue

        row = {'v1': v1, 'v2': v2}

        # ── Both quantitative → Pearson r ─────────────────────────────────
        if t1 == 'quantitative' and t2 == 'quantitative':
            r, pval = scipy.stats.pearsonr(s1, s2)
            row.update(test='pearson_r', stat=r, dof=len(df_) - 2, pval=pval,
                       descriptive=f"r={r:.3f}, n={len(df_)}")

        # ── Quantitative × binary → Welch t-test ──────────────────────────
        elif {t1, t2} == {'quantitative', 'binary'}:
            quant_s = s1 if t1 == 'quantitative' else s2
            cat_s   = s2 if t1 == 'quantitative' else s1
            levels  = sorted(cat_s.unique())
            g0 = quant_s[cat_s == levels[0]]
            g1 = quant_s[cat_s == levels[1]]
            if len(g0) < 2 or len(g1) < 2:
                continue
            t_res = scipy.stats.ttest_ind(g0, g1, equal_var=False)
            row.update(test='welch_t', stat=t_res.statistic,
                       dof=round(t_res.df, 1), pval=t_res.pvalue,
                       descriptive=_group_stats(quant_s, cat_s))

        # ── Quantitative × multicategory → one-way ANOVA ──────────────────
        elif (t1 == 'quantitative' and t2 == 'multicategory') or \
             (t1 == 'multicategory' and t2 == 'quantitative'):
            quant_s = s1 if t1 == 'quantitative' else s2
            cat_s   = s2 if t1 == 'quantitative' else s1
            groups  = [quant_s[cat_s == lev].values
                       for lev in sorted(cat_s.unique())
                       if (cat_s == lev).sum() >= 2]
            if len(groups) < 2:
                continue
            f_stat, pval = scipy.stats.f_oneway(*groups)
            k = len(groups)
            n = sum(len(g) for g in groups)
            row.update(test='anova_F', stat=f_stat,
                       dof=f"({k - 1}, {n - k})", pval=pval,
                       descriptive=_group_stats(quant_s, cat_s))

        # ── Both categorical → chi2 / proportions z-test ──────────────────
        else:
            crosstab = pd.crosstab(s1, s2)
            if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
                continue
            chi2_stat, pval, dof, _ = scipy.stats.chi2_contingency(crosstab)
            print("####\n", crosstab)

            if cattest == 'prop_ztest' and t1 == 'binary' and t2 == 'binary':
                levels_s1   = sorted(s1.unique())
                outcome_lev = sorted(s2.unique())[-1]   # treat highest level as "event"
                counts = [crosstab.at[lev, outcome_lev]
                          if outcome_lev in crosstab.columns else 0
                          for lev in levels_s1]
                nobs   = [crosstab.loc[lev].sum() for lev in levels_s1]
                z_stat, pval = proportions_ztest(count=counts, nobs=nobs,
                                                  value=None, alternative='two-sided')
                row.update(test='prop_ztest', stat=z_stat, dof=1, pval=pval)
                print(counts, nobs)
            else:
                row.update(test='chi2', stat=chi2_stat, dof=dof, pval=pval)

            row['descriptive'] = crosstab.to_string().replace('\n', ' | ')

        rows.append(row)

    cols = ['v1', 'v2', 'test', 'stat', 'dof', 'pval', 'descriptive']
    return (pd.DataFrame(rows, columns=cols)
              .sort_values('pval')
              .reset_index(drop=True))


# %% Descriptive statistics for clinical variables

#data.to_csv(os.path.join(config['output_models'], 'data.csv'), index=False)
data.response.describe()
["Resp=%i, N=%i" % (resp, np.sum(data.response == resp)) for resp in data.response.unique()]
['Resp=1, N=24', 'Resp=0, N=34']

# %% Run Univariate statistics

# data.Catatonie[data.response == 0].mean()
# data.Catatonie[data.response == 1].mean()

stats = univ_stats(data, vars1=['response'], vars2=config['clinical_vars'], cattest="prop_ztest")
stats.sort_values('pval', inplace=True)
stats.to_csv(os.path.join(config['output_models'], 'univariate_stats_v-2.csv'), index=False)

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

