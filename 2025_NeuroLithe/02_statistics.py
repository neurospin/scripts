"""
Univariate statistics for clinical variables.
"""


import pandas as pd
import numpy as np
import os
from config import data, config

from utils.stats_pairwise import pairwise_stats



# %% Descriptive statistics for clinical variables

#data.to_csv(os.path.join(config['output_models'], 'data.csv'), index=False)
data.response.describe()
["Resp=%i, N=%i" % (resp, np.sum(data.response == resp)) for resp in data.response.unique()]
['Resp=1, N=24', 'Resp=0, N=34']

# %% Run Univariate statistics

# data.Catatonie[data.response == 0].mean()
# data.Catatonie[data.response == 1].mean()

stats = pairwise_stats(data, vars1=['response'], vars2=config['clinical_vars'], cattest="prop_ztest")
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

