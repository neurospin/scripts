# -*- coding: utf-8 -*-
"""
Created on Fri Jul  18  2025

"""

import os

# Manipulate data
import numpy as np
import pandas as pd
import itertools
# Statistics
import scipy.stats
import statsmodels.api as sm
#import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
#from statsmodels.stats.stattools import jarque_bera


# Set Working Directory

config = dict(
    # Set the working directory
    working_directory='/home/ed203246/git/scripts/2025_NeuroLithe',
    # Set the path to the data file
    data_file='data/NeuroLithe_V1707.xlsx',
    # Set the path to save results
    output_models='models/'
)

os.chdir(config['working_directory'])


# %% Load data
nrows = 61 - 3
data = pd.read_excel(config['data_file'], sheet_name='Database', skiprows=2, nrows=nrows)
assert data.Patient.iloc[-1] == 'NEUROLITHE_058'
data.dtypes
# Display first few rows of the dataset
print(data.head())

# Response

response = \
    (data.rehospi_T2 == 0) & (data.Scolarite_T2 == 1) & (data.PSP_FONCTIONNEMENT >= 70)
    
response = \
    (data.rehospi_T2 == 0) & (data.Scolarite_T2 == 1)
response = response.astype(int)
data['response'] = response


# Input variables
input_vars_dict = dict(
    QI = ['QI<85', '85-115', 'QI>115'],
    Familiaux_psy=['actdpsy1', 'atcd_depression1', 'atcd_TH1', 'atcd_psychose1', 'atcd_TND1'],
    TND_DSM_V = ['DSM_MOT', 'DSM_TSA', 'DSM_TDAH', 'DSM_Tr App'],
    other = ['Catatonie'],
    TEMPSA= ['TEMPSA-C', 'TEMPSA-D', 'TEMPSA-I', 'TEMPSA-H', 'TEMPSA-A'],
    PQ16= ['PQ16-T', 'PQ16-A'],
    CDI=['CDI'],
    ASQ= ['ASQtot'],
    Atcd_trauma=['atcd_trauma']
)

input_vars = [v for set in input_vars_dict.values() for v in set]  # Flatten the list of input variables

# Check that all input variables are numeric
print(data[input_vars + ['response']].dtypes)
assert len(data[input_vars + ['response']].select_dtypes(include=np.number).columns) ==\
    len(input_vars + ['response'])  


# %% Utils
# ========

def is_categorical(x, levels=2):
    """
    Guess if x is categorical.
    Returns True if x is not float and has at most 'levels' unique values.
    """
    return pd.Series(x).nunique(dropna=True) <= levels


def univ_stats(data, cols1, cols2):

    res = list()

    for x, y in itertools.product(cols1, cols2):
        #y = 'Catatonie'
        df_ = data[[x, y]].dropna()
        # Check that all columns are numeric
        # df_.dtypes
        # df_.describe()
        if is_categorical(df_.values.ravel(), levels=2): # Chi2
            crosstab = pd.crosstab(df_[x], df_[y], rownames=[x], colnames=[y])
            stat, pval, dof, expected = scipy.stats.chi2_contingency(crosstab)
            test = "chi2"
            
        elif is_categorical(df_[y].values, levels=2): # two-sample t-test / y
            l0, l1 = np.unique(df_[y].values)
            ttest = scipy.stats.ttest_ind(df_.loc[df_[y] == l1, x], df_.loc[df_[y] == l0, x], equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            test = "ttest"
        
        elif is_categorical(df_[x].values, levels=2): # two-sample t-test / x
            l0, l1 = np.unique(df_[x].values)
            ttest = scipy.stats.ttest_ind(df_.loc[df_[x] == l1, y], df_.loc[df_[x] == l0, y], equal_var=False)
            stat, pval = ttest.statistic, ttest.pvalue
            test = "ttest"
        
        else:
            test = scipy.stats.pearsonr(df_[x], df_[y])
            stat, pval = test.statistic, test.pvalue
            test = "corr"
        
        res.append([x, y, stat, pval, test])

    res = pd.DataFrame(res, columns=['v1', 'v2', 'stat', 'pval', 'test'])
    res = res.sort_values( 'pval')
    return(res)


# data.Catatonie[data.response == 0].mean()
# data.Catatonie[data.response == 1].mean()

stats = univ_stats(data, cols1=['response'], cols2=input_vars)
stats.sort_values('pval', inplace=True)
stats.to_csv(os.path.join(config['output_models'], 'univariate_stats.csv'), index=False)

