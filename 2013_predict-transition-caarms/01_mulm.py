# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:23:37 2014

@author: edouard.duchesnay@cea.fr
"""

#import os.path
import os
import sys
import numpy as np
import pandas as pd

#from sklearn import preprocessing
import sklearn.feature_selection

WD = "/home/edouard/data/2013_predict-transition-caarms"
SRC = "/home/edouard/git/scripts/2013_predict-transition-caarms"

#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../lib'))
sys.path.append(SRC)

import IO
Xd, yd = IO.read_Xy(WD=WD)
Xd.PAS2gr[Xd.PAS2gr==1] = -1
Xd.PAS2gr[Xd.PAS2gr==2] = 1
Xd.CB_EXPO[Xd.CB_EXPO==0] = -1

X = np.asarray(Xd)
y = np.asarray(yd)

# mean / sd
mean = Xd.mean(axis=1)
std = Xd.std(axis=1)


f_stat, f_pval = sklearn.feature_selection.f_classif(X, y)
#chi2, chi2_pval = sklearn.feature_selection.chi2(X, y)

univ = pd.DataFrame(dict(var=Xd.columns, mean=mean, std=std,
                  f_stat=f_stat, f_pval=f_pval))#,
#                  chi2=chi2, chi2_pval=chi2_pval))
univ.to_csv(os.path.join(WD, "results", "univ_stats.csv"))

print univ
#      f_pval    f_stat      mean       std      var
#0   0.013310  7.098765  1.555556  1.783112   PAS2gr
#1   0.088194  3.148148  0.962963  1.159625  CB_EXPO
#2   0.043541  4.520697  2.111111  1.908147     @1.1
#3   0.283246  1.202639  2.333333  1.797434     @1.2
#4   0.408669  0.706215  2.555556  1.825742     @1.3
#5   0.364852  0.851852  2.592593  1.865873     @2.1
#6   0.147367  2.235772  2.592593  1.865873     @2.2
#7        NaN -0.569801  2.370370  1.445102     @3.1
#8   0.685077  0.168350  1.333333  1.921538     @3.2
#9   0.367663  0.841751  2.925926  1.465656     @3.3
#10       NaN -0.555556  2.444444  2.207214     @4.1
#11  0.284105  1.198257  2.185185  2.038797     @4.2
#12  0.108062  2.777778  3.037037  1.628626     @4.3
#13  0.457010  0.570776  2.370370  2.002847     @5.1
#14  0.275989  1.240391  2.666667  1.860521     @5.2
#15  0.608685  0.268817  2.259259  1.700511     @5.3
#16  0.038547  4.770277  2.555556  1.846688     @5.4
#17  0.531910  0.401817  2.592593  1.337600     @6.1
#18  0.089815  3.114478  3.111111  1.552500     @6.3
#19  0.556032  0.356125  1.814815  1.442141     @6.4
#20  0.474817  0.526507  1.000000  1.208941     @7.2
#21       NaN -0.170068  2.296296  1.877290     @7.3
#22  0.152851  2.173913  2.481481  2.172740     @7.4
#23  1.000000  0.000000  1.962963  1.786304     @7.5
#24  0.035072  4.966330  1.074074  1.639088     @7.6
#25  0.146252  2.248677  1.666667  1.818706     @7.7
#26       NaN -0.070028  2.444444  1.908147     @7.8



print univ[univ.f_pval <= 0.05]
#      f_pval    f_stat      mean       std     var
#0   0.013310  7.098765  1.555556  1.783112  PAS2gr
#2   0.043541  4.520697  2.111111  1.908147    @1.1
#16  0.038547  4.770277  2.555556  1.846688    @5.4
#24  0.035072  4.966330  1.074074  1.639088    @7.6


from sklearn.metrics import confusion_matrix

confusion_matrix(np.asarray(Xd.PAS2gr), y)
#array([[ 0,  0,  0],
#       [11,  2,  0],
#       [ 5,  9,  0]])
"""
pas2gr = as.factor(c(1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1))

transition = as.factor(c(0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
       0, 0, 0, 0))
table(pas, transition)
   transition
pas  0  1
  1 11  2
  2  5  9

chisq.test(pas, transition, correct=FALSE)

	Pearson's Chi-squared test

data:  pas and transition
X-squared = 6.6767, df = 1, p-value = 0.009768
"""

sklearn.feature_selection.chi2(np.asarray(Xd.PAS2gr), y)

np.unique(np.asarray(yd))
np.unique(np.asarray(Xd.PAS2gr))



