#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:42:05 2019

@author: anton
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import mannwhitneyu, ttest_ind, chisquare
import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.pyplot import figure
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats.stats import pearsonr
from sklearn import svm, metrics, linear_model
from statsmodels.stats.proportion import proportions_ztest

pheno = pd.read_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/Analysis_pooled_ROIs_pooled_phenotypes/data/phenotypes_SCHIZCONNECT_VIP_PRAGUE_BSNIP_BIOBD_ICAAR_START_ages_precis.tsv', sep='\t')

list(pheno)

pheno = pheno[pheno.irm == 'M0']

assert len(pheno[pheno.diagnosis == 'UHR-C']) == 27
assert len(pheno[pheno.diagnosis == 'UHR-NC']) == 53


#########################################################################################################################################
"""AGE"""
assert pheno[pheno.diagnosis == 'UHR-C'].true_age.mean() == 20.752021700002544
assert pheno[pheno.diagnosis == 'UHR-C'].true_age.std() == 1.9116881455276264
assert pheno[pheno.diagnosis == 'UHR-NC'].true_age.mean() == 22.828044890421406
assert pheno[pheno.diagnosis == 'UHR-NC'].true_age.std() == 3.0241729835124733
age1 = pheno[pheno.diagnosis == 'UHR-C'].true_age
age1.iloc[19] = 21.0 # age of 21 taken from pheno.age column
age1 = np.array(age1)
age2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].true_age)
ttest_ind(age1, age2)
# Ttest_indResult(statistic=-3.246324893391699, pvalue=0.0017239415950476903)

T_array = age1
NT_array = age2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['age','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.age[df.status == 'UHR-C'], df.age[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    age = df.age.values
    age = age[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[age,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.age[df_permuted.status == 'UHR-C'], df_permuted.age[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm #  0.0029997000299970002
#########################################################################################################################################
"""SEX"""
assert len(pheno.sex[pheno.diagnosis == 'UHR-C'][pheno.sex == 1.0]) == 8
assert len(pheno.sex[pheno.diagnosis == 'UHR-C'][pheno.sex == 0.0]) == 19
assert len(pheno.sex[pheno.diagnosis == 'UHR-NC'][pheno.sex == 1.0]) == 23
assert len(pheno.sex[pheno.diagnosis == 'UHR-NC'][pheno.sex == 0.0]) == 30
chisquare([29.63, 43.40])
# Power_divergenceResult(statistic=2.5963699849376964, pvalue=0.10710878766622178)

#########################################################################################################################################
"""MADRS"""
assert pheno[pheno.diagnosis == 'UHR-C'].MADRS.mean() == 20.962962962962962
assert pheno[pheno.diagnosis == 'UHR-C'].MADRS.std() == 7.465208954164963
assert pheno[pheno.diagnosis == 'UHR-NC'].MADRS.mean() == 21.50943396226415
assert pheno[pheno.diagnosis == 'UHR-NC'].MADRS.std() == 10.315452638395382

madrs1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].MADRS)
madrs2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].MADRS)
ttest_ind(madrs1, madrs2)
# Ttest_indResult(statistic=-0.2442827752151795, pvalue=0.8076532596839414)

T_array = madrs1
NT_array = madrs2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.8121187881211879

#########################################################################################################################################
"""PANSS total"""
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_total.mean() == 72.81481481481481
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_total.std() == 14.884263856211046
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_total.mean() == 64.45283018867924
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_total.std() == 17.682636026226245

panss1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].PANSS_total)
panss2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].PANSS_total)
ttest_ind(panss1, panss2)
# Ttest_indResult(statistic=2.104896086925813, pvalue=0.03851988465873452)

T_array = panss1
NT_array = panss2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.0374962503749625

#########################################################################################################################################
"""PANSS positive"""
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_positive.mean() == 15.62962962962963
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_positive.std() == 5.62376925440408
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_positive.mean() == 11.88679245283019
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_positive.std() == 4.444791149192721

pos1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].PANSS_positive)
pos2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].PANSS_positive)
ttest_ind(pos1, pos2)
# Ttest_indResult(statistic=3.250739355420391, pvalue=0.0017005724139933073)

T_array = pos1
NT_array = pos2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.0016998300169983002

#########################################################################################################################################
"""PANSS negative"""
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_negative.mean() == 16.51851851851852
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_negative.std() == 6.559610508901426
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_negative.mean() == 15.18867924528302
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_negative.std() == 6.920028356360269

neg1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].PANSS_negative)
neg2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].PANSS_negative)
ttest_ind(neg1, neg2)
# Ttest_indResult(statistic=0.8268688398875617, pvalue=0.41083380446679585)

T_array = neg1
NT_array = neg2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.40315968403159685

#########################################################################################################################################
"""PANSS desorganisation"""
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_desorganisation.mean() == 6.592592592592593
assert pheno[pheno.diagnosis == 'UHR-C'].PANSS_desorganisation.std() == 2.5154791447426725
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_desorganisation.mean() == 5.283018867924528
assert pheno[pheno.diagnosis == 'UHR-NC'].PANSS_desorganisation.std() == 2.178367053291967

des1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].PANSS_desorganisation)
des2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].PANSS_desorganisation)
ttest_ind(des1, des2)
# Ttest_indResult(statistic=2.4120530406409264, pvalue=0.01821241865294439)

T_array = des1
NT_array = des2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.0168983101689831

#########################################################################################################################################
"""SOFAS"""
assert pheno[pheno.diagnosis == 'UHR-C'].SOFAS.mean() == 49.03703703703704
assert pheno[pheno.diagnosis == 'UHR-C'].SOFAS.std() == 9.724993500023462
assert pheno[pheno.diagnosis == 'UHR-NC'].SOFAS.mean() == 49.03846153846154
assert pheno[pheno.diagnosis == 'UHR-NC'].SOFAS.std() == 9.805883664895891

sof1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].SOFAS)
sof2 = list(pheno[pheno.diagnosis == 'UHR-NC'].SOFAS)
del sof2[51]
sof2 = np.array(sof2)
ttest_ind(sof1, sof2)
# Ttest_indResult(statistic=-0.0006141213831294517, pvalue=0.9995115903391332)

T_array = sof1
NT_array = sof2
assert len(T_array) == 27
assert len(NT_array) == 52
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(52)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 1.0

#########################################################################################################################################
"""Eq Chlorpromazine"""
pheno.Eq_Chlorpromazine.fillna(0, inplace=True)
assert len(pheno[(pheno.diagnosis == 'UHR-C') & (pheno.Eq_Chlorpromazine != 0)]) == 6
6/27 # 0.2222222222222222
assert len(pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.Eq_Chlorpromazine != 0)]) == 7
7/53 # 0.1320754716981132
assert pheno[(pheno.diagnosis == 'UHR-C') & (pheno.Eq_Chlorpromazine != 0)].Eq_Chlorpromazine.mean() == 87.66666666666667
assert pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.Eq_Chlorpromazine != 0)].Eq_Chlorpromazine.mean() == 94.78571428571429
assert pheno[(pheno.diagnosis == 'UHR-C') & (pheno.Eq_Chlorpromazine != 0)].Eq_Chlorpromazine.std() == 44.432720675946314
assert pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.Eq_Chlorpromazine != 0)].Eq_Chlorpromazine.std() == 43.98092118829969

chisquare([22,13.2])
# Power_divergenceResult(statistic=2.2, pvalue=0.13801073756865542)

chlo1 = np.array(pheno[pheno.diagnosis == 'UHR-C'].Eq_Chlorpromazine)
chlo2 = np.array(pheno[pheno.diagnosis == 'UHR-NC'].Eq_Chlorpromazine)
ttest_ind(chlo1, chlo2)
# Ttest_indResult(statistic=0.7773695480408613, pvalue=0.43929216250202474)

T_array = chlo1
NT_array = chlo2
assert len(T_array) == 27
assert len(NT_array) == 53
T_array = T_array[:,np.newaxis]
NT_array = NT_array[:,np.newaxis]
C_array = np.ones(27)
NC_array = np.zeros(53)
C_array = C_array[:,np.newaxis]
NC_array = NC_array[:,np.newaxis]
array_C = np.c_[T_array,C_array]
array_NC = np.c_[NT_array,NC_array]
data_array = np.r_[array_C,array_NC]
columns = ['variable','status']
df = pd.DataFrame(data_array, columns=columns)
df.status = df.status.map({1.0:'UHR-C', 0.0:'UHR-NC'})

# Permutation: simulate the null hypothesis

nperm = 10000
perms = np.zeros(nperm + 1)
p = np.zeros(nperm + 1)
perms[0], p[0] = ttest_ind(df.variable[df.status == 'UHR-C'], df.variable[df.status == 'UHR-NC'])

# je veux permuter aléatoirement l'assignation à chaque groupe puis faire le wilcoxon
    # je permute le statut converteur
    # je les mets dans un nouveau tableau
    # je fais le mann whitney
    # je stocke la statistique U et p

for i in range(1,nperm):
    variable = df.variable.values
    variable = variable[:,np.newaxis]
    perm = np.random.permutation(df.status)
    perm = perm[:,np.newaxis]
    perm_assign = np.c_[variable,perm]
    df_permuted = pd.DataFrame(perm_assign, columns=columns)
    perms[i], p[i] = ttest_ind(df.variable[df_permuted.status == 'UHR-C'], df_permuted.variable[df_permuted.status == 'UHR-NC'])

# Two-tailed empirical p-value
pval_perm = np.sum(p <= p[0])/p.shape[0]
pval_perm # 0.44925507449255075


#########################################################################################################################################
"""SUBSTANCE USE"""

from collections import Counter

#alcohol_last_month (nb of drinks): 1 = 0 ; 2 = 1-2 ; 3 = 3-9 ; 4 = 10-39 ; 5 = >40
pheno.alcohol_last_month.fillna(0, inplace=True)
pheno.alcohol_last_month = pheno.alcohol_last_month.map({0:1, '1':1, '2':2, '3':3, '4':4, '5':5, 'ND':1})
assert len(pheno[(pheno.diagnosis == 'UHR-C') & (pheno.alcohol_last_month != 1)]) == 13
13/27 # 0.48148148148148145
assert len(pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.alcohol_last_month != 1)]) == 19
19/53 # 0.3584905660377358
alcohol = pheno[pheno.diagnosis == 'UHR-C'].alcohol_last_month
Counter(alcohol).keys() # dict_keys([3, 1, 2, 4])
Counter(alcohol).values() # dict_values([6, 14, 5, 2])
alcohol1 = pheno[pheno.diagnosis == 'UHR-NC'].alcohol_last_month
Counter(alcohol1).keys() # dict_keys([1, 3, 4, 2, 5])
Counter(alcohol1).values() # dict_values([34, 9, 5, 4, 1])

chisquare([48.15, 35.85])
# Power_divergenceResult(statistic=1.8010714285714278, pvalue=0.179583018342859)


#tobacco_last_mont (nb of cig/day): 1 = 0 ; 2 = >1/jr ; 3 = 1-5 ; 4 = 6-10 ; 5 = 11-20 ; 6 = >20
pheno.tobacco_last_month.fillna(0, inplace=True)
pheno.tobacco_last_month = pheno.tobacco_last_month.map({0:1, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, 'ND':1})
assert len(pheno[(pheno.diagnosis == 'UHR-C') & (pheno.tobacco_last_month != 1)]) == 16
16/27 # 0.5925925925925926
assert len(pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.tobacco_last_month != 1)]) == 20
20/53 # 0.37735849056603776
tobacco = pheno[pheno.diagnosis == 'UHR-C'].tobacco_last_month
Counter(tobacco).keys() # dict_keys([4, 5, 1, 3, 6, 2])
Counter(tobacco).values() # dict_values([3, 4, 11, 4, 3, 2])
tobacco1 = pheno[pheno.diagnosis == 'UHR-NC'].tobacco_last_month
Counter(tobacco1).keys() # dict_keys([1, 4, 3, 5, 2])
Counter(tobacco1).values() # dict_values([33, 6, 6, 7, 1])

chisquare([59.30, 37.74])
# Power_divergenceResult(statistic=4.7901236603462465, pvalue=0.028623371544310508)

#cannabis_last_month (nb of times one smoked): 1 = 0 ; 2 = 1-2 ; 3 = 3-9 ; 4 = 10-39 ; 5 = >40
pheno.cannabis_last_month.fillna(0, inplace=True)
pheno.cannabis_last_month = pheno.cannabis_last_month.map({0.0:1, 1.0:1, 2.0:2, 3.0:3, 4.0:4, 5.5:5, 'ND':1})
assert len(pheno[(pheno.diagnosis == 'UHR-C') & (pheno.cannabis_last_month != 1)]) == 13
13/27 # 0.48148148148148145
assert len(pheno[(pheno.diagnosis == 'UHR-NC') & (pheno.cannabis_last_month != 1)]) == 12
12/53 # 0.22641509433962265
cannabis = pheno[pheno.diagnosis == 'UHR-C'].cannabis_last_month
Counter(cannabis).keys() # dict_keys([2, 3, 1, 4])
Counter(cannabis).values() # dict_values([4, 4, 14, 5])
cannabis1 = pheno[pheno.diagnosis == 'UHR-NC'].cannabis_last_month
Counter(cannabis1).keys() # dict_keys([1, 4, 2])
Counter(cannabis1).values() # dict_values([41, 7, 5])

chisquare([48.15, 22.64])
# Power_divergenceResult(statistic=9.192825257804774, pvalue=0.0024296558270055606)




# pour chi2 trend test for proportions on R
trend = pheno[pheno.diagnosis.isin(['UHR-C','UHR-NC'])][['diagnosis','alcohol_last_month','tobacco_last_month','cannabis_last_month']]
trend.diagnosis = trend.diagnosis.map({'UHR-C':'UHR_C','UHR-NC':'UHR_NC'})
diagnosis = pd.get_dummies(trend.diagnosis)
trend = trend[['alcohol_last_month','tobacco_last_month','cannabis_last_month']]
trend = pd.concat([diagnosis, trend], axis=1)
set(trend.alcohol_last_month) # {1, 2, 3, 4, 5}
set(trend.tobacco_last_month) # {1, 2, 3, 4, 5, 6}
set(trend.cannabis_last_month) # {1, 2, 3, 4}


trend.to_csv('/home/anton/Dropbox/PhD/Article_Age_epigenetique_et_cerebral/2019_Brain_age/substance_use_ICAARSTART.csv',index=False)

"""
alcohol

	Chi-squared Test for Trend in Proportions

data:  case.vector1 out of total.vector1 ,
 using scores: 1 2 3 4 5
X-squared = 0.090557, df = 1, p-value = 0.7635
"""

"""
tobacco

	Chi-squared Test for Trend in Proportions

data:  case.vector2 out of total.vector2 ,
 using scores: 1 2 3 4 5 6
X-squared = 3.4472, df = 1, p-value = 0.06336
"""

"""
cannabis

	Chi-squared Test for Trend in Proportions

data:  case.vector3 out of total.vector3 ,
 using scores: 1 2 3 4
X-squared = 3.7939, df = 1, p-value = 0.05144

"""


