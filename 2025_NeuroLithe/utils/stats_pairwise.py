import pandas as pd
import numpy as np
import scipy.stats
from statsmodels.stats.proportion import proportions_ztest

import itertools

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


def pairwise_stats(data: pd.DataFrame, vars1: list, vars2: list,
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