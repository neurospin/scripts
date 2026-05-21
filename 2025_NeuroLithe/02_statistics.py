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

# # %% Additional Statistics with propotion test 

# v1, v2 ='DSM_TDAH', 'response'
# df_ = data[[v1, v2]].dropna()
# ct = pd.crosstab(df_[v1], df_[v2], rownames=[v1], colnames=[v2])
# no_resp, no_sum = ct.iloc[0, 1], ct.iloc[0, :].sum()
# yes_resp, yes_sum = ct.iloc[1, 1], ct.iloc[1, :].sum()
# print("Prop No resp %.3f vs Prop Resp %.3f" % (no_resp / no_sum, yes_resp / yes_sum))


# proportions_ztest(count=[no_resp, yes_resp], nobs=[no_sum, yes_sum], value=None, alternative='two-sided')
# # (np.float64(-2.015767098021012), np.float64(0.0438243355622758))

# proportions_ztest(count=[no_resp, yes_resp], nobs=[no_sum, yes_sum], value=None, alternative='smaller',)
# # (np.float64(-2.015767098021012), np.float64(0.0219121677811379))

# %% Other Descriptive statistics

Demographie = ['AGE', 'Sexe']
Famille		= ['parent_div', 'parent_dcd', 'descola', 'scola_nle']
Scolarite = ['niveau_scol', 'scola_nle', 'descola', 'harcel_scol']

Antécédents = ['depression', 'manie_hypomanie', 'mixte', 'Tr anxieux', 'TOC', 'TCA', 'TND', 'TS', 'IDS', 'CAM', 'nb_hospi psy', 'nb_ant_APA', 'nb_ant_ATD', 'nb_ant_TR', 'nb ligne tt', 'ttt_MPH', 'cannabis_T1', 'actdpsy1', 'atcd_depression1', 'atcd_TH1', 'atcd_psychose1', 'atcd_TND1', 'atcd_ttt_Li', 'atcd_psy2']
Qi = ['QI<85', '85-115', 'QI>115']

Diagnostics = ['PEP_nonthymique', 'PEP_thymique', 'TH_nonpsychotique', 'TDDE', 'TDAH avec RF']
Traitements_T1 = ['TR', 'ATD', 'APA', 'Eq_CPZ', 'Metformine']
Traitements_T2 = ['Metformine_T2', 'MPH_T2', 'TR_T2', 'ATD_T2', 'APA_T2']

# 
Demographie_stat = pairwise_stats(data, vars1=['response'], vars2=Demographie, cattest="prop_ztest", max_cat_unique=5)
Demographie_stat.insert(0, 'Domain', 'Demographie')
Diagnostics_stat = pairwise_stats(data, vars1=['response'], vars2=Diagnostics, cattest="prop_ztest", max_cat_unique=5)
Diagnostics_stat.insert(0, 'Domain', 'Diagnostics')
Scolarite_stat = pairwise_stats(data, vars1=['response'], vars2=Scolarite, cattest="prop_ztest", max_cat_unique=5)
Scolarite_stat.insert(0, 'Domain', 'Scolarite')
Antécédents_stat = pairwise_stats(data, vars1=['response'], vars2=Antécédents, cattest="prop_ztest", max_cat_unique=5)
Antécédents_stat.insert(0, 'Domain', 'Antécédents')
Qi_stat = pairwise_stats(data, vars1=['response'], vars2=Qi, cattest="prop_ztest", max_cat_unique=5)
Qi_stat.insert(0, 'Domain', 'Qi')
Traitements_T1_stat = pairwise_stats(data, vars1=['response'], vars2=Traitements_T1, cattest="prop_ztest", max_cat_unique=5)
Traitements_T1_stat.insert(0, 'Domain', 'Traitements_T1')
Traitements_T2_stat = pairwise_stats(data, vars1=['response'], vars2=Traitements_T2, cattest="prop_ztest")
Traitements_T2_stat.insert(0, 'Domain', 'Traitements_T2')

all_stats = pd.concat([
    Demographie_stat, Diagnostics_stat, Scolarite_stat,
    Antécédents_stat, Qi_stat, Traitements_T1_stat, Traitements_T2_stat,
], ignore_index=True)

excel_path = os.path.join(config['output_models'], 'reports/descriptive_statistics.xlsx')
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    all_stats.to_excel(writer, sheet_name="all_domains", index=False)
    for domain, df in all_stats.groupby("Domain", sort=False):
        df.to_excel(writer, sheet_name=domain[:31], index=False)
print(f"Saved {excel_path}")

# %%
